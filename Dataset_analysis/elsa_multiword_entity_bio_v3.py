"""
=============================================================================
ELSA Dataset - Multi-Word Entity Detection Using BIO Tagging  [v4]
=============================================================================
Dataset  : ELSA_Dataset_-_ELSA_10K.csv
Language : Bangla (Bengali) product reviews
Task     : Detect named entities using BIO scheme, identify multi-word entities
           and generate a full analysis report as a .txt file

BIO Scheme Used:
  B-PRODUCT  -> Beginning token of a product/entity name
  I-PRODUCT  -> Inside (continuation) token of a multi-word entity
  O          -> Outside (non-entity) token

Multi-word entity logic:
  The dataset uses _NE_ prefix to tag entity tokens.
  e.g. "_NE_ডেটল সাবান" -> 'ডেটল' = B-PRODUCT, 'সাবান' = I-PRODUCT
  e.g. "_NE_প্রোডাক্ট"   -> 'প্রোডাক্ট' = B-PRODUCT (single-token entity)

Fixes applied in v2:
  BUG 1  – Same-word repetition noise removed (G1 guard)
  BUG 2  – Interior-punctuation artifact guard (G2 guard)
  LIMIT  – Incomplete continuation whitelist extended + comma-list guard (G3)

NEW in v3:
  FEATURE – Multi-Word Entity Sentence-Level Tracking
             For every unique multi-word entity, the report now lists:
               • Each Sentence/Review ID in which it appears
               • How many times it occurs within that sentence
             This allows manual verification by finding the exact sentence
             in the dataset and checking the entity occurrence directly.

NEW in v4:
  FEATURE 1 – Sentiment Analysis Report for Multi-Word Entity Sentences only
               Saved as: multiword_sentiment_report.txt
               Contains:
                 • Overall Positive / Neutral / Negative counts & percentages
                 • Per-entity breakdown across all sentiments
                 • Per-sentiment top entities table
                 • Sample sentences per sentiment

  FEATURE 2 – JSON Export (multiword_entities.json)
               One record per multi-word-entity sentence with:
                 • Sequential ID (1, 2, 3 …) — no _NE_ tags
                 • original_csv_id  (real ID from the CSV)
                 • sentence         (clean text, _NE_ removed)
                 • sentiment        (label + numeric code)
                 • bio_tags         (per-token index + tag)
                 • multi_word_entities  (text, tokens, BIO start/end index)
                 • all_entity_spans     (single + multi for full context)

  FEATURE 3 – XML Export (multiword_entities.xml)
               Same structure as JSON, serialised as well-formed XML.
=============================================================================
"""

import pandas as pd
import re
import os
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FILENAME = "ELSA_Dataset - ELSA_10K.csv"

def _find_input_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filename,
        os.path.join(os.getcwd(), filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "datasets", filename),
        os.path.join(os.getcwd(), "datasets", filename),
        os.path.join(os.path.expanduser("~"), filename),
        os.path.join("/mnt/user-data/uploads", filename),
        # Also try with hyphen variant
        os.path.join(os.getcwd(), filename.replace(" - ", "_-_")),
        os.path.join(script_dir, filename.replace(" - ", "_-_")),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return filename

INPUT_FILE  = _find_input_file(_DEFAULT_FILENAME)
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
OUTPUT_TXT  = os.path.join(OUTPUT_DIR, "elsa_multiword_entity_report_v3.txt")
OUTPUT_BIO  = os.path.join(OUTPUT_DIR, "elsa_bio_tagged_sentences_v3.txt")

# ── NEW in v4 ────────────────────────────────────────────────────────────────
OUTPUT_SENTIMENT_TXT = os.path.join(OUTPUT_DIR, "multiword_sentiment_report.txt")
OUTPUT_JSON          = os.path.join(OUTPUT_DIR, "multiword_entities.json")
OUTPUT_XML           = os.path.join(OUTPUT_DIR, "multiword_entities.xml")

# Known multi-word product continuation words in Bangla reviews.
PRODUCT_SECOND_TOKENS = {
    'সাবান', 'ওয়াশ', 'অয়েল', 'ওয়েল', 'লোশন', 'ক্রিম', 'শ্যাম্পু',
    'টাচ', 'মাস্ক', 'ট্রিমার', 'ব্রাশ', 'ব্যাগ', 'তেল', 'বার',
    'ফোম', 'পাউডার', 'জেল', 'সিরাম', 'টোনার', 'স্প্রে', 'বাম',
    'বাটার', 'ড্রপ', 'ট্যাবলেট', 'ক্যাপসুল', 'সিরাপ', 'সোপ',
    'oil', 'wash', 'cream', 'soap', 'mask', 'lotion',
    'shampoo', 'powder', 'gel', 'spray', 'serum', 'toner',
    'balm', 'butter', 'foam', 'bar', 'drop', 'tablet',
    'capsule', 'syrup', 'trimmer', 'brush',
}

SENTIMENT_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_token(token):
    """Remove trailing punctuation only (used for final entity text display)."""
    return re.sub(r'[।\.?!,;:\(\)\"\'৷\-]+$', '', token).strip()


# Bangla inflectional suffixes that attach to product words in reviews.
# NOTE: _E_MATRA (U+09C7, ে) is the DEPENDENT vowel sign — different from
#       the independent vowel এ (U+098F). Genitive suffix in 'সাবানের' is
#       ে+র, NOT এ+র. Using the wrong codepoint causes silent match failure.
_E_MATRA = '\u09c7'   # ে  dependent vowel sign E
_BANGLA_SUFFIXES = [
    'গুলো', 'গুলি', 'গুলা',         # plural classifiers  (longest first)
    'খানা', 'খানি',                   # unit classifiers
    'টা',   'টি',   'টো',            # definiteness markers
    _E_MATRA + 'র',                   # ের  genitive (সাবানের → সাবান)
    'কে',                             # dative / accusative
    'তে',                             # locative
    'র',                              # bare genitive (vowel-final stems)
    'ও',                              # additive
]


def normalize_token(token):
    """
    Full normalization for matching a raw dataset token against the
    PRODUCT_SECOND_TOKENS whitelist (or building a clean entity stem).

    Steps applied in order:
      1. Split on any mid-token sentence-boundary character (।, ., !, ?, ;)
         and keep only the part BEFORE it.
         Fixes: 'ওয়াশ।রিকমন্ডেড।' -> 'ওয়াশ'
      2. Strip trailing ASCII/Bangla punctuation (comma, colon, quotes).
         Fixes: 'ওয়াশ,' -> 'ওয়াশ'
      3. Strip Bangla inflectional suffixes (longest-first, one suffix only).
         Fixes: 'সাবানের' -> 'সাবান'
               'ক্রিমটি' -> 'ক্রিম'
               'শ্যাম্পুটা' -> 'শ্যাম্পু'
               'ওয়াশের'   -> 'ওয়াশ'
    """
    stem = re.split(r'[।\.?!;৷]', token)[0].strip()
    stem = re.sub(r'[,:\(\)\"\']+$', '', stem).strip()
    for suf in _BANGLA_SUFFIXES:
        if stem.endswith(suf) and len(stem) > len(suf) + 1:
            stem = stem[:-len(suf)]
            break
    return stem


def tokenize(text):
    return text.split()

def is_punctuation_only(token):
    return bool(re.match(r'^[।\.?!,;:\(\)\"\'৷\-\s❤️?★☆✓✗@#%&*+=/<>~^]+$', token))

def extract_bio_tags(tagged_review, original_review=""):
    if not isinstance(tagged_review, str):
        return []

    raw_tokens = tokenize(tagged_review)
    result = []
    i = 0

    while i < len(raw_tokens):
        tok = raw_tokens[i]

        if tok.startswith('_NE_'):
            entity_word = tok[4:]
            if not entity_word:
                i += 1
                continue

            # Use normalize_token for the B-token stem:
            # handles mid-punct split AND suffix stripping
            # (clean_token is kept only for display/span text output)
            entity_clean = normalize_token(entity_word)
            result.append((entity_word, 'B-PRODUCT'))
            i += 1

            # G2: if the raw entity word (before normalization) contains a
            # sentence-boundary punctuation in the MIDDLE, it is a tagging
            # artifact spanning two sentences — never look ahead.
            if re.search(r'[।\.?!;৷]', entity_word[1:]):
                continue

            accumulated_span_words = {entity_clean.lower()}
            while i < len(raw_tokens):
                next_tok = raw_tokens[i]
                if next_tok.startswith('_NE_'):
                    break

                # normalize_token handles:
                #   - mid-punct split: 'ওয়াশ।রিকমন্ডেড।' -> 'ওয়াশ'
                #   - trailing punct:  'ওয়াশ,'            -> 'ওয়াশ'
                #   - Bangla suffixes: 'সাবানের'           -> 'সাবান'
                #                      'ক্রিমটা'           -> 'ক্রিম'
                next_clean = normalize_token(next_tok).lower()

                if next_clean not in {w.lower() for w in PRODUCT_SECOND_TOKENS}:
                    break

                # G1: same-word repetition guard
                if next_clean in accumulated_span_words:
                    break

                # G3: comma-list guard on original review
                if isinstance(original_review, str) and original_review:
                    comma_pattern = re.escape(entity_clean) + r'\s*[,،]'
                    if re.search(comma_pattern, original_review):
                        break

                result.append((next_tok, 'I-PRODUCT'))
                accumulated_span_words.add(next_clean)
                i += 1
        else:
            if not is_punctuation_only(tok) and tok.strip():
                result.append((tok, 'O'))
            i += 1

    return result


def get_entity_spans(bio_pairs):
    spans = []
    i = 0
    while i < len(bio_pairs):
        token, tag = bio_pairs[i]
        if tag == 'B-PRODUCT':
            # normalize_token gives the clean stem (no suffix, no mid-punct)
            # used both for the span text and for deduplication keys
            span_tokens = [normalize_token(token)]
            j = i + 1
            while j < len(bio_pairs) and bio_pairs[j][1] == 'I-PRODUCT':
                span_tokens.append(normalize_token(bio_pairs[j][0]))
                j += 1
            spans.append({
                'tokens'   : span_tokens,
                'text'     : ' '.join(span_tokens),
                'start_idx': i,
                'end_idx'  : j - 1,
                'is_multi' : len(span_tokens) > 1
            })
            i = j
        else:
            i += 1
    return spans


# ── NEW in v4 ────────────────────────────────────────────────────────────────
def clean_review(tagged_review):
    """Return plain sentence text with all _NE_ prefixes removed."""
    if not isinstance(tagged_review, str):
        return ""
    return re.sub(r'_NE_', '', tagged_review).strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(input_file):
    print(f"[INFO] Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[INFO] Total rows loaded: {len(df)}")

    rows_bio        = []
    bio_output      = []
    all_entities    = []
    all_single      = []
    all_multi       = []

    total_tokens          = 0
    total_entity_tokens   = 0
    total_b_tags          = 0
    total_i_tags          = 0
    total_o_tags          = 0
    single_entity_count   = 0
    multi_entity_count    = 0
    sentences_with_entity = 0
    sentences_no_entity   = 0

    sentiment_stats = defaultdict(lambda: {
        'total': 0, 'single': 0, 'multi': 0, 'entities': []
    })

    # NEW in v3: track for every unique multi-word entity which sentence IDs
    # it appears in and how many times per sentence.
    # Structure: { entity_text: { sentence_id: count } }
    multi_entity_sentence_map = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        doc_id     = row['ID']
        review     = str(row['REVIEW'])
        tagged     = str(row['ENTITY_TAGGED_REVIEW'])
        sentiment  = int(row['ENTITY_SENTIMENT'])
        sent_label = SENTIMENT_MAP.get(sentiment, 'Unknown')

        bio_pairs = extract_bio_tags(tagged, review)
        spans     = get_entity_spans(bio_pairs)

        b_count = sum(1 for _, t in bio_pairs if t == 'B-PRODUCT')
        i_count = sum(1 for _, t in bio_pairs if t == 'I-PRODUCT')
        o_count = sum(1 for _, t in bio_pairs if t == 'O')
        total_tokens        += len(bio_pairs)
        total_entity_tokens += b_count + i_count
        total_b_tags        += b_count
        total_i_tags        += i_count
        total_o_tags        += o_count

        row_single = [s for s in spans if not s['is_multi']]
        row_multi  = [s for s in spans if s['is_multi']]
        single_entity_count += len(row_single)
        multi_entity_count  += len(row_multi)

        if spans:
            sentences_with_entity += 1
        else:
            sentences_no_entity += 1

        for s in spans:
            all_entities.append(s['text'])
            if s['is_multi']:
                all_multi.append(s['text'])
                # v3: record which sentence this multi-word entity appears in
                multi_entity_sentence_map[s['text']][doc_id] += 1
            else:
                all_single.append(s['text'])

        sentiment_stats[sent_label]['total'] += 1
        sentiment_stats[sent_label]['single'] += len(row_single)
        sentiment_stats[sent_label]['multi']  += len(row_multi)
        sentiment_stats[sent_label]['entities'].extend([s['text'] for s in spans])

        rows_bio.append({
            'ID'           : doc_id,
            'SENTIMENT'    : sent_label,
            'REVIEW'       : review,
            'BIO_PAIRS'    : bio_pairs,
            'ENTITY_SPANS' : spans,
            'B_COUNT'      : b_count,
            'I_COUNT'      : i_count,
            'O_COUNT'      : o_count,
            'SINGLE_ENTS'  : len(row_single),
            'MULTI_ENTS'   : len(row_multi),
        })

        bio_output.append(f"=== ID: {doc_id} | Sentiment: {sent_label} ===")
        bio_output.append(f"REVIEW : {review}")
        bio_output.append("BIO TAGS:")
        bio_output.append(f"  {'TOKEN':<30} {'TAG'}")
        bio_output.append(f"  {'-'*30} {'-'*12}")
        for token, tag in bio_pairs:
            bio_output.append(f"  {token:<30} {tag}")
        if spans:
            bio_output.append("ENTITIES FOUND:")
            for s in spans:
                kind = 'MULTI-WORD' if s['is_multi'] else 'SINGLE-WORD'
                bio_output.append(f"  [{kind}]  '{s['text']}'  ({len(s['tokens'])} token(s))")
        else:
            bio_output.append("ENTITIES FOUND: None")
        bio_output.append("")

    entity_freq    = Counter(all_entities)
    single_freq    = Counter(all_single)
    multi_freq     = Counter(all_multi)
    total_entities = single_entity_count + multi_entity_count

    stats = {
        'total_rows'                 : len(df),
        'total_tokens'               : total_tokens,
        'total_entity_tokens'        : total_entity_tokens,
        'total_b_tags'               : total_b_tags,
        'total_i_tags'               : total_i_tags,
        'total_o_tags'               : total_o_tags,
        'total_entities'             : total_entities,
        'single_entity_count'        : single_entity_count,
        'multi_entity_count'         : multi_entity_count,
        'sentences_with_entity'      : sentences_with_entity,
        'sentences_no_entity'        : sentences_no_entity,
        'unique_entities'            : len(entity_freq),
        'unique_single'              : len(single_freq),
        'unique_multi'               : len(multi_freq),
        'entity_freq'                : entity_freq,
        'single_freq'                : single_freq,
        'multi_freq'                 : multi_freq,
        'sentiment_stats'            : sentiment_stats,
        'rows_bio'                   : rows_bio,
        'multi_entity_sentence_map'  : multi_entity_sentence_map,  # NEW v3
    }

    return df, stats, bio_output


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(stats, output_txt_path, bio_output, output_bio_path):
    s = stats
    total          = s['total_rows']
    total_tok      = s['total_tokens']
    ent_tok        = s['total_entity_tokens']
    total_ent      = s['total_entities']
    single_cnt     = s['single_entity_count']
    multi_cnt      = s['multi_entity_count']
    pct_single     = (single_cnt / total_ent * 100) if total_ent else 0
    pct_multi      = (multi_cnt  / total_ent * 100) if total_ent else 0
    pct_ent_sents  = (s['sentences_with_entity'] / total * 100) if total else 0
    pct_no_sents   = (s['sentences_no_entity']   / total * 100) if total else 0
    pct_b          = (s['total_b_tags'] / total_tok * 100) if total_tok else 0
    pct_i          = (s['total_i_tags'] / total_tok * 100) if total_tok else 0
    pct_o          = (s['total_o_tags'] / total_tok * 100) if total_tok else 0

    lines = []
    H1 = "=" * 80
    H2 = "-" * 80
    H3 = "~" * 60

    def section(title):
        lines.append("")
        lines.append(H1)
        lines.append(f"  {title}")
        lines.append(H1)

    def subsection(title):
        lines.append("")
        lines.append(H3)
        lines.append(f"  {title}")
        lines.append(H3)

    # ── HEADER ──────────────────────────────────────────────────────────────
    lines.append(H1)
    lines.append("  ELSA DATASET — MULTI-WORD ENTITY DETECTION REPORT  [v3 — WITH SENTENCE IDs]")
    lines.append("  BIO (B-PRODUCT / I-PRODUCT / O) Tagging Analysis")
    lines.append("  Language: Bangla (Bengali) | Domain: E-commerce Product Reviews")
    lines.append(H1)

    # ── SECTION 1: DATASET OVERVIEW ─────────────────────────────────────────
    section("SECTION 1: DATASET OVERVIEW")
    lines.append(f"  Total Reviews (Sentences)  : {total:>8,}")
    lines.append(f"  Total Tokens (all)         : {total_tok:>8,}")
    lines.append(f"  Total Entity Tokens (B+I)  : {ent_tok:>8,}")
    lines.append(f"  Total Non-Entity Tokens (O): {s['total_o_tags']:>8,}")
    lines.append("")
    lines.append(f"  Sentiment Distribution:")
    for label, data in sorted(s['sentiment_stats'].items()):
        pct = data['total'] / total * 100
        lines.append(f"    {label:<12}: {data['total']:>6,}  ({pct:.1f}%)")

    # ── SECTION 2: BIO TAG STATISTICS ───────────────────────────────────────
    section("SECTION 2: BIO TAG STATISTICS")
    lines.append(f"  {'Tag':<15} {'Count':>10} {'% of All Tokens':>18}")
    lines.append(f"  {'-'*15} {'-'*10} {'-'*18}")
    lines.append(f"  {'B-PRODUCT':<15} {s['total_b_tags']:>10,} {pct_b:>17.2f}%")
    lines.append(f"  {'I-PRODUCT':<15} {s['total_i_tags']:>10,} {pct_i:>17.2f}%")
    lines.append(f"  {'O':<15} {s['total_o_tags']:>10,} {pct_o:>17.2f}%")
    lines.append(f"  {'TOTAL':<15} {total_tok:>10,} {'100.00%':>18}")
    lines.append("")
    lines.append("  Explanation of Tags:")
    lines.append("    B-PRODUCT : First/only token of a product/entity name")
    lines.append("    I-PRODUCT : Continuation token (2nd, 3rd word of multi-word entity)")
    lines.append("    O         : Non-entity token (regular word)")
    lines.append("")
    lines.append("  v2/v3 Fixes Applied (affect I-PRODUCT / multi-word counts only):")
    lines.append("    G1 – Same-word repetition noise removed")
    lines.append("         e.g. _NE_সাবান সাবান  →  single-word 'সাবান' (not 'সাবান সাবান')")
    lines.append("    G2 – Interior sentence-boundary punctuation guard")
    lines.append("         e.g. _NE_পণ্য।ইতিপূর্বেও ব্যাগ  →  single-word only")
    lines.append("    G3 – Comma-list guard (uses original review text)")
    lines.append("         e.g. 'তেল, শ্যাম্পু'  →  two separate single-word entities")

    # ── SECTION 3: ENTITY SUMMARY ───────────────────────────────────────────
    section("SECTION 3: ENTITY SUMMARY")
    lines.append(f"  Total Entity Spans Detected    : {total_ent:>8,}")
    lines.append(f"  Single-Word Entities (B only)  : {single_cnt:>8,}  ({pct_single:.1f}%)")
    lines.append(f"  Multi-Word Entities  (B+I)     : {multi_cnt:>8,}  ({pct_multi:.1f}%)")
    lines.append("")
    lines.append(f"  Unique Entity Strings (total)  : {s['unique_entities']:>8,}")
    lines.append(f"  Unique Single-Word Entities    : {s['unique_single']:>8,}")
    lines.append(f"  Unique Multi-Word Entities     : {s['unique_multi']:>8,}")
    lines.append("")
    lines.append(f"  Sentences WITH at least 1 entity: {s['sentences_with_entity']:>7,}  ({pct_ent_sents:.1f}%)")
    lines.append(f"  Sentences WITH NO entity        : {s['sentences_no_entity']:>7,}  ({pct_no_sents:.1f}%)")

    # ── SECTION 4: TOP SINGLE-WORD ENTITIES ─────────────────────────────────
    section("SECTION 4: TOP 30 SINGLE-WORD ENTITIES (by frequency)")
    lines.append(f"  {'Rank':<6} {'Entity':<35} {'Count':>8} {'% of Single':>14}")
    lines.append(f"  {'-'*6} {'-'*35} {'-'*8} {'-'*14}")
    for rank, (ent, cnt) in enumerate(s['single_freq'].most_common(30), 1):
        pct_e = cnt / single_cnt * 100 if single_cnt else 0
        lines.append(f"  {rank:<6} {ent:<35} {cnt:>8,} {pct_e:>13.2f}%")

    # ── SECTION 5: ALL MULTI-WORD ENTITIES ──────────────────────────────────
    section("SECTION 5: ALL MULTI-WORD ENTITIES DETECTED")
    lines.append(f"  Total unique multi-word entities: {s['unique_multi']}")
    lines.append("")
    lines.append("  NOTE: 'IDs' column lists all Sentence/Review IDs where the entity")
    lines.append("        was detected. Use these to locate the exact row in the CSV")
    lines.append("        for manual verification. Format: ID(xN) means the entity")
    lines.append("        appears N times within that single sentence.")
    lines.append("")
    lines.append(f"  {'Rank':<6} {'Multi-Word Entity':<40} {'Count':>8} {'% of Multi':>12}  IDs")
    lines.append(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*12}  {'-'*50}")
    mw_sent_map = s['multi_entity_sentence_map']
    for rank, (ent, cnt) in enumerate(s['multi_freq'].most_common(), 1):
        pct_e = cnt / multi_cnt * 100 if multi_cnt else 0
        # Build IDs string: sorted IDs, annotate with (xN) if count > 1
        sent_id_map = mw_sent_map[ent]
        id_parts = []
        for sid in sorted(sent_id_map.keys()):
            sc = sent_id_map[sid]
            id_parts.append(str(sid) if sc == 1 else f"{sid}(x{sc})")
        ids_str = ", ".join(id_parts)
        # First line: stats + start of IDs
        base_line = f"  {rank:<6} {ent:<40} {cnt:>8,} {pct_e:>11.2f}%  "
        # Wrap IDs across continuation lines at ~60 chars to keep readable
        id_chunks = []
        chunk = ""
        for part in id_parts:
            candidate = (chunk + ", " + part) if chunk else part
            if len(candidate) > 60 and chunk:
                id_chunks.append(chunk)
                chunk = part
            else:
                chunk = candidate
        if chunk:
            id_chunks.append(chunk)
        lines.append(base_line + (id_chunks[0] if id_chunks else "—"))
        indent = " " * len(base_line)
        for extra_chunk in id_chunks[1:]:
            lines.append(indent + extra_chunk)

    # ── SECTION 6: MULTI-WORD ENTITY ANALYSIS ───────────────────────────────
    section("SECTION 6: MULTI-WORD ENTITY — DEEP ANALYSIS")

    subsection("6.1  Token Length Distribution of Multi-Word Entities")
    len_counter = Counter()
    for rb in s['rows_bio']:
        for span in rb['ENTITY_SPANS']:
            if span['is_multi']:
                len_counter[len(span['tokens'])] += 1
    lines.append(f"  {'Token Length':<15} {'Count':>8} {'%':>10}")
    lines.append(f"  {'-'*15} {'-'*8} {'-'*10}")
    for length in sorted(len_counter):
        cnt = len_counter[length]
        pct_l = cnt / multi_cnt * 100 if multi_cnt else 0
        lines.append(f"  {length:<15} {cnt:>8,} {pct_l:>9.2f}%")

    subsection("6.2  Multi-Word Entity vs Single-Word Entity by Sentiment")
    lines.append(f"  {'Sentiment':<12} {'Total Reviews':>15} {'Single Ents':>13} {'Multi Ents':>12} {'Multi %':>10}")
    lines.append(f"  {'-'*12} {'-'*15} {'-'*13} {'-'*12} {'-'*10}")
    for label in ['Positive', 'Neutral', 'Negative']:
        data = s['sentiment_stats'][label]
        t = data['total']
        sn = data['single']
        mu = data['multi']
        total_ents_here = sn + mu
        pct_m = mu / total_ents_here * 100 if total_ents_here else 0
        lines.append(f"  {label:<12} {t:>15,} {sn:>13,} {mu:>12,} {pct_m:>9.1f}%")

    subsection("6.3  Most Frequent Multi-Word Entities per Sentiment")
    for label in ['Positive', 'Neutral', 'Negative']:
        data = s['sentiment_stats'][label]
        ents = [e for e in data['entities'] if ' ' in e]
        freq = Counter(ents)
        lines.append(f"\n  [{label}] Top 10 Multi-Word Entities:")
        if freq:
            for rank, (ent, cnt) in enumerate(freq.most_common(10), 1):
                lines.append(f"    {rank:>2}. {ent:<35} — {cnt} occurrences")
        else:
            lines.append("    (none found)")

    subsection("6.4  Relationship: Multi-Word Entity → Single-Token Disambiguation")
    lines.append("""
  In the ELSA Bangla dataset, multi-word entities follow the same principle
  described in the MEID paper (AAAI 2020):

  PATTERN:
    Multi-word entity  →  "ডেটল সাবান"  (Dettol Soap)   [B-PRODUCT I-PRODUCT]
    Single-word entity →  "সাবান"        (soap — ambiguous alone)  [B-PRODUCT]

  WHY IT MATTERS:
    • "সাবান" alone could refer to any soap brand.
    • "ডেটল সাবান" uniquely identifies the product as Dettol brand soap.
    • "ফেস ওয়াশ" (face wash) clarifies it is a face cleanser product.
    • "প্যানটিন শ্যাম্পু" tells us both brand (Pantene) + product type.

  DISAMBIGUATION RULE:
    When a single-token entity like "সাবান" appears alone in a review,
    check if a multi-token entity containing "সাবান" exists elsewhere in
    the document or corpus → use its context to resolve the full product name.
  """)

    # ── SECTION 7: SAMPLE BIO-TAGGED REVIEWS ────────────────────────────────
    section("SECTION 7: SAMPLE BIO-TAGGED REVIEWS (10 Examples)")

    samples_shown = 0
    for rb in s['rows_bio']:
        if samples_shown >= 10:
            break
        if not rb['ENTITY_SPANS']:
            continue
        lines.append(f"  Review ID  : {rb['ID']}")
        lines.append(f"  Sentiment  : {rb['SENTIMENT']}")
        lines.append(f"  Review     : {rb['REVIEW'][:120]}")
        lines.append(f"  {'TOKEN':<30} {'TAG'}")
        lines.append(f"  {'-'*30} {'-'*12}")
        for token, tag in rb['BIO_PAIRS']:
            lines.append(f"  {token:<30} {tag}")
        lines.append("  Entities:")
        for span in rb['ENTITY_SPANS']:
            kind = "MULTI-WORD" if span['is_multi'] else "SINGLE"
            lines.append(f"    → [{kind}] '{span['text']}'")
        lines.append("")
        samples_shown += 1

    # ── SECTION 8: MULTI-WORD ENTITY EXAMPLES WITH CONTEXT ──────────────────
    section("SECTION 8: MULTI-WORD ENTITY EXAMPLES WITH FULL SENTENCE CONTEXT")
    shown = 0
    for rb in s['rows_bio']:
        if shown >= 20:
            break
        multi_spans = [sp for sp in rb['ENTITY_SPANS'] if sp['is_multi']]
        if not multi_spans:
            continue
        lines.append(f"  ID {rb['ID']} [{rb['SENTIMENT']}]")
        lines.append(f"  Review  : {rb['REVIEW'][:150]}")
        for sp in multi_spans:
            tokens_detail = ' + '.join([f"'{t}'" for t in sp['tokens']])
            tags_detail = 'B-PRODUCT' + ' + I-PRODUCT' * (len(sp['tokens']) - 1)
            lines.append(f"  Entity  : '{sp['text']}'")
            lines.append(f"  Tokens  : {tokens_detail}")
            lines.append(f"  BIO     : {tags_detail}")
        lines.append("")
        shown += 1

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 9 (NEW v3): MULTI-WORD ENTITIES WITH SENTENCE IDs
    # ────────────────────────────────────────────────────────────────────────
    section("SECTION 9: MULTI-WORD ENTITY — SENTENCE ID OCCURRENCE MAP  [NEW in v3]")
    lines.append("  PURPOSE:")
    lines.append("    This section allows MANUAL VERIFICATION of every unique multi-word")
    lines.append("    entity. For each entity you can:")
    lines.append("      1. Find the exact Sentence/Review IDs listed below")
    lines.append("      2. Open the CSV, filter on the ID column to that row")
    lines.append("      3. Read the REVIEW text and confirm the entity is correctly detected")
    lines.append("      4. The 'Count in Sentence' column shows how many times the entity")
    lines.append("         appears within that single review (usually 1, sometimes >1)")
    lines.append("")
    lines.append("  HOW TO USE:")
    lines.append("    → Entities are sorted by total occurrence frequency (highest first)")
    lines.append("    → Sentence IDs are sorted ascending for easy CSV lookup")
    lines.append("    → A '(x2)', '(x3)' marker means the entity appears that many times")
    lines.append("      within the SAME sentence/review")
    lines.append("")

    multi_entity_sentence_map = s['multi_entity_sentence_map']

    # Sort entities by total frequency
    sorted_multi_entities = s['multi_freq'].most_common()

    for rank, (entity_text, total_occ) in enumerate(sorted_multi_entities, 1):
        sentence_id_map = multi_entity_sentence_map[entity_text]
        num_sentences   = len(sentence_id_map)
        sorted_ids      = sorted(sentence_id_map.items(), key=lambda x: x[0])

        lines.append(f"  {'─'*76}")
        lines.append(f"  RANK #{rank:<4}  Entity: '{entity_text}'")
        lines.append(f"           Total Occurrences : {total_occ}")
        lines.append(f"           Appears in        : {num_sentences} unique sentence(s)")
        lines.append(f"           Avg per sentence  : {total_occ/num_sentences:.2f}")
        lines.append("")
        lines.append(f"  {'Sentence ID':<20} {'Occurrences in that Sentence':>30}")
        lines.append(f"  {'-'*20} {'-'*30}")

        for sent_id, count in sorted_ids:
            count_str = f"{count}" if count == 1 else f"{count}  (appears x{count} in this sentence)"
            lines.append(f"  {str(sent_id):<20} {count_str}")

        lines.append("")

    # ── SECTION 10: COMPLETE ENTITY FREQUENCY TABLE ─────────────────────────
    section("SECTION 10: COMPLETE ENTITY FREQUENCY TABLE (All Entities, Sorted)")
    lines.append(f"  {'Rank':<6} {'Entity':<40} {'Count':>8} {'Type':<12} {'% of Total Ents':>16}")
    lines.append(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*12} {'-'*16}")
    for rank, (ent, cnt) in enumerate(s['entity_freq'].most_common(), 1):
        etype = 'MULTI-WORD' if ' ' in ent else 'SINGLE'
        pct_e = cnt / total_ent * 100 if total_ent else 0
        lines.append(f"  {rank:<6} {ent:<40} {cnt:>8,} {etype:<12} {pct_e:>15.3f}%")

    # ── SECTION 11: SUMMARY INSIGHTS ────────────────────────────────────────
    section("SECTION 11: SUMMARY INSIGHTS & CONCLUSIONS")
    multi_pct_of_total = multi_cnt / total_ent * 100 if total_ent else 0
    lines.append(f"""
  1. SCALE
     The ELSA 10K dataset contains {total:,} product reviews in Bangla.
     A total of {total_ent:,} entity spans were detected across {s['sentences_with_entity']:,}
     sentences ({pct_ent_sents:.1f}% of all reviews contain at least one entity).

  2. SINGLE vs MULTI-WORD ENTITIES
     • Single-word entities: {single_cnt:,} ({pct_single:.1f}%)
     • Multi-word entities : {multi_cnt:,}  ({pct_multi:.1f}%)
     Multi-word entities make up {multi_pct_of_total:.1f}% of all detected entities.

  3. MOST COMMON SINGLE-WORD ENTITIES
     The most frequent single-word product mentions are generic terms:
     'প্রোডাক্ট' (product), 'বই' (book), 'পণ্য' (goods), 'জিনিস' (item).
     These are highly AMBIGUOUS without multi-word context.

  4. MOST COMMON MULTI-WORD ENTITIES
     • 'ফেস ওয়াশ'   (Face Wash)    — cosmetic product
     • 'ডেটল সাবান'  (Dettol Soap)  — brand + product type
     • 'অলিভ অয়েল'  (Olive Oil)    — descriptor + product
     These multi-word forms DISAMBIGUATE the product category clearly.

  5. BIO TAG COVERAGE
     B-PRODUCT tags: {s['total_b_tags']:,} ({pct_b:.2f}% of tokens)
     I-PRODUCT tags: {s['total_i_tags']:,} ({pct_i:.2f}% of tokens)
     O tags        : {s['total_o_tags']:,} ({pct_o:.2f}% of tokens)

  6. SENTIMENT CORRELATION
     Multi-word entities appear across all sentiment classes, indicating
     that reviewers use specific product names regardless of satisfaction.
     Positive reviews tend to have more entity mentions overall.

  7. NEW v3: SENTENCE ID MAP (Section 9)
     Every unique multi-word entity now lists the exact Sentence/Review IDs
     in which it was detected. Use these IDs to open the CSV, locate that
     row, and manually verify the entity detection is correct.
     Entries marked '(x2)', '(x3)' etc. mean the entity appears multiple
     times within a single review — worth special attention during QA.

  8. IMPLICATION FOR NER MODELS
     Following the MEID (AAAI 2020) approach:
     → Multi-word entities like 'ডেটল সাবান' should receive HIGHER
       attention weights when disambiguating single-token occurrences
       of 'সাবান' or 'ডেটল' elsewhere in the same document.
     → An auxiliary MEC (Multi-token Entity Classification) task can
       pre-label SUB vs NSUB tokens before the main NER decoding step.
  """)

    lines.append(H1)
    lines.append("  END OF REPORT  [v3]")
    lines.append(H1)

    # ── WRITE MAIN REPORT ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] Report written to: {output_txt_path}")

    # ── WRITE BIO TAGGED FILE ────────────────────────────────────────────────
    with open(output_bio_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bio_output))
    print(f"[INFO] BIO-tagged sentences written to: {output_bio_path}")

    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NEW v4 — SENTIMENT REPORT FOR MULTI-WORD ENTITY SENTENCES ONLY
# ─────────────────────────────────────────────────────────────────────────────

def generate_multiword_sentiment_report(stats, output_path):
    """
    Write a sentiment analysis report that covers ONLY sentences which contain
    at least one multi-word entity (B-PRODUCT + I-PRODUCT span).

    Sections:
      1. Overall Positive / Neutral / Negative count & percentage
      2. Multi-word entity frequency table split by sentiment
      3. Per-sentiment top-entity tables
      4. Up to 5 sample sentences per sentiment
    """
    rows_bio = stats['rows_bio']

    # Filter to multi-word-entity sentences only
    mw_rows = [r for r in rows_bio if r['MULTI_ENTS'] > 0]
    total   = len(mw_rows)

    sentiment_counts       = Counter(r['SENTIMENT'] for r in mw_rows)
    entity_freq            = Counter()
    per_sentiment_entities = defaultdict(Counter)

    for r in mw_rows:
        for sp in r['ENTITY_SPANS']:
            if sp['is_multi']:
                entity_freq[sp['text']] += 1
                per_sentiment_entities[r['SENTIMENT']][sp['text']] += 1

    H1 = "=" * 80
    H2 = "-" * 80
    H3 = "~" * 60
    lines = []

    lines.append(H1)
    lines.append("  ELSA DATASET — MULTI-WORD ENTITY SENTENCES: SENTIMENT ANALYSIS REPORT  [v4]")
    lines.append("  Scope    : ONLY sentences that contain at least one multi-word entity (B+I)")
    lines.append("  Language : Bangla (Bengali) | Domain: E-commerce Product Reviews")
    lines.append(H1)

    # ── SECTION 1: OVERALL DISTRIBUTION ──────────────────────────────────────
    lines.append("")
    lines.append("  SECTION 1: OVERALL SENTIMENT DISTRIBUTION")
    lines.append(H2)
    lines.append(f"  Total sentences with multi-word entities : {total:,}")
    lines.append("")
    lines.append(f"  {'Sentiment':<12} {'Count':>8} {'Percentage':>12}")
    lines.append(f"  {'-'*12} {'-'*8} {'-'*12}")
    for label in ['Positive', 'Neutral', 'Negative']:
        cnt = sentiment_counts.get(label, 0)
        pct = cnt / total * 100 if total else 0
        lines.append(f"  {label:<12} {cnt:>8,} {pct:>11.1f}%")

    # ── SECTION 2: ENTITY FREQUENCY × SENTIMENT ──────────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  SECTION 2: MULTI-WORD ENTITY FREQUENCY ACROSS ALL SENTIMENTS")
    lines.append(H2)
    lines.append(f"  {'Rank':<6} {'Multi-Word Entity':<40} {'Total':>8}"
                 f" {'Positive':>10} {'Neutral':>8} {'Negative':>10}")
    lines.append(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
    for rank, (ent, total_cnt) in enumerate(entity_freq.most_common(), 1):
        pos_c = per_sentiment_entities['Positive'][ent]
        neu_c = per_sentiment_entities['Neutral'][ent]
        neg_c = per_sentiment_entities['Negative'][ent]
        lines.append(f"  {rank:<6} {ent:<40} {total_cnt:>8,}"
                     f" {pos_c:>10,} {neu_c:>8,} {neg_c:>10,}")

    # ── SECTION 3: PER-SENTIMENT BREAKDOWN ───────────────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  SECTION 3: PER-SENTIMENT BREAKDOWN")
    lines.append(H2)
    for label in ['Positive', 'Neutral', 'Negative']:
        cnt = sentiment_counts.get(label, 0)
        pct = cnt / total * 100 if total else 0
        lines.append("")
        lines.append(f"  [{label.upper()}]  {cnt:,} sentences  "
                     f"({pct:.1f}% of multi-word-entity sentences)")
        lines.append(f"  {H3}")
        ents_here = per_sentiment_entities[label]
        if ents_here:
            lines.append(f"  Top Multi-Word Entities in {label} sentences:")
            lines.append(f"  {'Rank':<6} {'Entity':<40} {'Count':>8}")
            lines.append(f"  {'-'*6} {'-'*40} {'-'*8}")
            for rank, (ent, c) in enumerate(ents_here.most_common(20), 1):
                lines.append(f"  {rank:<6} {ent:<40} {c:>8,}")
        else:
            lines.append("  (no multi-word entities found)")

    # ── SECTION 4: SAMPLE SENTENCES ──────────────────────────────────────────
    lines.append("")
    lines.append("")
    lines.append("  SECTION 4: SAMPLE SENTENCES PER SENTIMENT (up to 5 each)")
    lines.append(H2)
    for label in ['Positive', 'Neutral', 'Negative']:
        shown = 0
        lines.append("")
        lines.append(f"  ── {label.upper()} EXAMPLES ──")
        for r in mw_rows:
            if r['SENTIMENT'] != label:
                continue
            if shown >= 5:
                break
            clean = clean_review(r['REVIEW'])
            lines.append(f"  ID {r['ID']}: {clean[:120]}")
            for sp in r['ENTITY_SPANS']:
                if sp['is_multi']:
                    lines.append(f"    → Multi-Word Entity: '{sp['text']}'  "
                                 f"({len(sp['tokens'])} tokens)")
            shown += 1

    lines.append("")
    lines.append(H1)
    lines.append("  END OF REPORT  [v4]")
    lines.append(H1)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] Sentiment report written to: {output_path}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# NEW v4 — JSON + XML EXPORT FOR MULTI-WORD ENTITY SENTENCES
# ─────────────────────────────────────────────────────────────────────────────

def _indent_xml(elem, level=0):
    """Add pretty-print indentation to an ElementTree in-place."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def generate_json_xml(stats, json_path, xml_path):
    """
    For every sentence that contains at least one multi-word entity, write:

      JSON record fields:
        id                  – sequential integer starting at 1 (no _NE_)
        original_csv_id     – real ID column value from the CSV
        sentence            – clean review text (_NE_ prefixes removed)
        sentiment           – { label, code }
        bio_tags            – list of { token_index, token, tag }
        multi_word_entities – list of { entity_text, tokens,
                                        token_count, bio_start_idx, bio_end_idx }
        all_entity_spans    – list of { text, is_multi, tokens }

      The XML mirrors the same structure as well-formed XML elements.
    """
    rows_bio = stats['rows_bio']
    mw_rows  = [r for r in rows_bio if r['MULTI_ENTS'] > 0]

    # ── Build shared record list ──────────────────────────────────────────────
    records = []
    for seq_id, r in enumerate(mw_rows, 1):
        bio_index_entries = [
            {"token_index": pos, "token": tok, "tag": tag}
            for pos, (tok, tag) in enumerate(r['BIO_PAIRS'])
        ]
        multi_ent_entries = [
            {
                "entity_text":   sp['text'],
                "tokens":        sp['tokens'],
                "token_count":   len(sp['tokens']),
                "bio_start_idx": sp['start_idx'],
                "bio_end_idx":   sp['end_idx'],
            }
            for sp in r['ENTITY_SPANS'] if sp['is_multi']
        ]
        all_span_entries = [
            {"text": sp['text'], "is_multi": sp['is_multi'], "tokens": sp['tokens']}
            for sp in r['ENTITY_SPANS']
        ]
        records.append({
            "id":               seq_id,
            "original_csv_id":  int(r['ID']),
            "sentence":         clean_review(r['REVIEW']),
            "sentiment": {
                "label": r['SENTIMENT'],
                "code":  {"Negative": 0, "Neutral": 1, "Positive": 2}.get(r['SENTIMENT'], -1)
            },
            "bio_tags":             bio_index_entries,
            "multi_word_entities":  multi_ent_entries,
            "all_entity_spans":     all_span_entries,
        })

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON written to         : {json_path}")

    # ── XML ───────────────────────────────────────────────────────────────────
    root = ET.Element("multiword_entity_sentences")

    for rec in records:
        sent_el = ET.SubElement(root, "sentence")
        sent_el.set("id",              str(rec["id"]))
        sent_el.set("original_csv_id", str(rec["original_csv_id"]))

        ET.SubElement(sent_el, "text").text = rec["sentence"]

        senti_el = ET.SubElement(sent_el, "sentiment")
        ET.SubElement(senti_el, "label").text = rec["sentiment"]["label"]
        ET.SubElement(senti_el, "code").text  = str(rec["sentiment"]["code"])

        bio_el = ET.SubElement(sent_el, "bio_tags")
        for entry in rec["bio_tags"]:
            tok_el = ET.SubElement(bio_el, "token")
            tok_el.set("index", str(entry["token_index"]))
            tok_el.set("tag",   entry["tag"])
            tok_el.text = entry["token"]

        mw_el = ET.SubElement(sent_el, "multi_word_entities")
        for sp in rec["multi_word_entities"]:
            ent_el = ET.SubElement(mw_el, "entity")
            ET.SubElement(ent_el, "text").text          = sp["entity_text"]
            ET.SubElement(ent_el, "token_count").text   = str(sp["token_count"])
            ET.SubElement(ent_el, "bio_start_idx").text = str(sp["bio_start_idx"])
            ET.SubElement(ent_el, "bio_end_idx").text   = str(sp["bio_end_idx"])
            toks_el = ET.SubElement(ent_el, "tokens")
            for t in sp["tokens"]:
                ET.SubElement(toks_el, "t").text = t

        all_el = ET.SubElement(sent_el, "all_entity_spans")
        for sp in rec["all_entity_spans"]:
            sp_el = ET.SubElement(all_el, "span")
            sp_el.set("is_multi", str(sp["is_multi"]).lower())
            ET.SubElement(sp_el, "text").text = sp["text"]
            toks_el = ET.SubElement(sp_el, "tokens")
            for t in sp["tokens"]:
                ET.SubElement(toks_el, "t").text = t

    _indent_xml(root)
    tree = ET.ElementTree(root)
    with open(xml_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)
    print(f"[INFO] XML written to          : {xml_path}")

    return len(records)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, stats, bio_output = process_dataset(INPUT_FILE)
    total_lines = generate_report(stats, OUTPUT_TXT, bio_output, OUTPUT_BIO)

    # ── NEW v4: sentiment report + JSON + XML ─────────────────────────────────
    mw_sent_count  = generate_multiword_sentiment_report(stats, OUTPUT_SENTIMENT_TXT)
    mw_json_count  = generate_json_xml(stats, OUTPUT_JSON, OUTPUT_XML)

    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    print(f"  Total reviews            : {stats['total_rows']:,}")
    print(f"  Total tokens             : {stats['total_tokens']:,}")
    print(f"  Total B-PRODUCT tags     : {stats['total_b_tags']:,}")
    print(f"  Total I-PRODUCT tags     : {stats['total_i_tags']:,}")
    print(f"  Total O tags             : {stats['total_o_tags']:,}")
    print(f"  Total entity spans       : {stats['total_entities']:,}")
    print(f"  Single-word entities     : {stats['single_entity_count']:,}")
    print(f"  Multi-word entities      : {stats['multi_entity_count']:,}")
    print(f"  Unique multi-word ents   : {stats['unique_multi']:,}")
    print(f"  Report lines written     : {total_lines:,}")
    print(f"  Multi-word-entity sents  : {mw_sent_count:,}  [v4]")
    print(f"  JSON / XML records       : {mw_json_count:,}  [v4]")
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_TXT}")
    print(f"    2. {OUTPUT_BIO}")
    print(f"    3. {OUTPUT_SENTIMENT_TXT}   [NEW v4]")
    print(f"    4. {OUTPUT_JSON}            [NEW v4]")
    print(f"    5. {OUTPUT_XML}             [NEW v4]")
    print("\n[DONE]")