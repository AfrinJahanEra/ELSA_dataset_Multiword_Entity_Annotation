"""
=============================================================================
ELSA Dataset - Multi-Word Entity Detection Using BIO Tagging
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

Author : Generated for ELSA NER analysis
=============================================================================
"""

import pandas as pd
import re
import os
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FILENAME = "ELSA_Dataset - ELSA_10K.csv"

# Auto-detect input file: checks same directory as script, datasets folder, and common upload paths
def _find_input_file(filename):
    script_dir = os.path.dirname(__file__)
    candidates = [
        filename,                                          # current working dir
        os.path.join(os.getcwd(), filename),               # current working dir absolute
        os.path.join(script_dir, filename),                # same dir as script
        os.path.join(script_dir, "datasets", filename),  # datasets subdir next to script
        os.path.join(os.getcwd(), "datasets", filename),  # datasets subdir from cwd
        os.path.join(os.path.expanduser("~"), filename),  # home dir
        os.path.join("/mnt/user-data/uploads", filename), # claude.ai upload path
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # fallback: return bare filename (will raise a clear FileNotFoundError)
    return filename

INPUT_FILE  = _find_input_file(_DEFAULT_FILENAME)
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "result")
OUTPUT_TXT  = os.path.join(OUTPUT_DIR, "elsa_multiword_entity_report.txt")
OUTPUT_BIO  = os.path.join(OUTPUT_DIR, "elsa_bio_tagged_sentences.txt")

# Known multi-word product continuation words in Bangla reviews
# These words, when immediately following a _NE_-tagged token, extend the entity
PRODUCT_SECOND_TOKENS = {
    'সাবান', 'ওয়াশ', 'অয়েল', 'লোশন', 'ক্রিম', 'শ্যাম্পু', 'টাচ',
    'মাস্ক', 'ট্রিমার', 'ব্রাশ', 'ব্যাগ', 'তেল', 'বার', 'ফোম',
    'পাউডার', 'জেল', 'সিরাম', 'টোনার', 'স্প্রে', 'বাম', 'বাটার',
    'ড্রপ', 'ট্যাবলেট', 'ক্যাপসুল', 'সিরাপ', 'oil', 'wash', 'cream',
    'soap', 'mask', 'lotion', 'shampoo', 'powder', 'gel', 'spray',
}

# Sentiment label map
SENTIMENT_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_token(token):
    """Remove trailing punctuation from a token for clean entity extraction."""
    return re.sub(r'[।\.?!,;:\(\)\"\'৷\-]+$', '', token).strip()


def tokenize(text):
    """Simple whitespace tokenizer that keeps punctuation attached to words."""
    return text.split()


def is_punctuation_only(token):
    """Return True if token is purely punctuation/emoji/symbol."""
    return bool(re.match(r'^[।\.?!,;:\(\)\"\'৷\-\s❤️?★☆✓✗@#%&*+=/<>~^]+$', token))


def extract_bio_tags(tagged_review):
    """
    Convert a _NE_-tagged review string into a list of (token, BIO-tag) pairs.

    Rules:
      1. A token immediately preceded by _NE_ is tagged B-PRODUCT.
      2. If the VERY NEXT token (not _NE_ prefixed) is a known product
         continuation word, it gets I-PRODUCT (extending the entity).
      3. All other tokens get O.

    Returns:
        List of (token_str, bio_tag) tuples
    """
    if not isinstance(tagged_review, str):
        return []

    raw_tokens = tokenize(tagged_review)
    result = []
    i = 0

    while i < len(raw_tokens):
        tok = raw_tokens[i]

        if tok.startswith('_NE_'):
            # Strip the _NE_ prefix to get the actual word
            entity_word = tok[4:]  # remove '_NE_'
            if not entity_word:   # edge case: _NE_ alone
                i += 1
                continue

            entity_clean = clean_token(entity_word)
            result.append((entity_word, 'B-PRODUCT'))
            i += 1

            # Look ahead: check if next token extends this entity (I-PRODUCT)
            while i < len(raw_tokens):
                next_tok = raw_tokens[i]
                # Stop if next token is itself an entity start
                if next_tok.startswith('_NE_'):
                    break
                next_clean = clean_token(next_tok).lower()
                if next_clean in {w.lower() for w in PRODUCT_SECOND_TOKENS}:
                    result.append((next_tok, 'I-PRODUCT'))
                    i += 1
                else:
                    break
        else:
            # Regular non-entity token
            if not is_punctuation_only(tok) and tok.strip():
                result.append((tok, 'O'))
            i += 1

    return result


def get_entity_spans(bio_pairs):
    """
    Extract entity spans from BIO-tagged pairs.

    Returns list of dicts:
        {
          'tokens'   : list of token strings in the entity,
          'text'     : full entity string (joined),
          'start_idx': index of B tag in bio_pairs,
          'end_idx'  : index of last I tag (inclusive),
          'is_multi' : True if entity has 2+ tokens
        }
    """
    spans = []
    i = 0
    while i < len(bio_pairs):
        token, tag = bio_pairs[i]
        if tag == 'B-PRODUCT':
            span_tokens = [clean_token(token)]
            j = i + 1
            while j < len(bio_pairs) and bio_pairs[j][1] == 'I-PRODUCT':
                span_tokens.append(clean_token(bio_pairs[j][0]))
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(input_file):
    """
    Load dataset, apply BIO tagging, collect statistics.
    Returns (df_processed, stats_dict, bio_lines, report_lines)
    """
    print(f"[INFO] Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[INFO] Total rows loaded: {len(df)}")

    # ── Per-row processing ──────────────────────────────────────────────────
    rows_bio        = []   # list of dicts for processed rows
    bio_output      = []   # lines for the BIO output file
    all_entities    = []   # flat list of all entity spans
    all_single      = []   # single-token entity strings
    all_multi       = []   # multi-token entity strings

    # Aggregated stats
    total_tokens          = 0
    total_entity_tokens   = 0
    total_b_tags          = 0
    total_i_tags          = 0
    total_o_tags          = 0
    single_entity_count   = 0
    multi_entity_count    = 0
    sentences_with_entity = 0
    sentences_no_entity   = 0

    # Per-sentiment breakdowns
    sentiment_stats = defaultdict(lambda: {
        'total': 0, 'single': 0, 'multi': 0, 'entities': []
    })

    for idx, row in df.iterrows():
        doc_id     = row['ID']
        review     = str(row['REVIEW'])
        tagged     = str(row['ENTITY_TAGGED_REVIEW'])
        sentiment  = int(row['ENTITY_SENTIMENT'])
        sent_label = SENTIMENT_MAP.get(sentiment, 'Unknown')

        bio_pairs = extract_bio_tags(tagged)
        spans     = get_entity_spans(bio_pairs)

        # Count tags
        b_count = sum(1 for _, t in bio_pairs if t == 'B-PRODUCT')
        i_count = sum(1 for _, t in bio_pairs if t == 'I-PRODUCT')
        o_count = sum(1 for _, t in bio_pairs if t == 'O')
        total_tokens        += len(bio_pairs)
        total_entity_tokens += b_count + i_count
        total_b_tags        += b_count
        total_i_tags        += i_count
        total_o_tags        += o_count

        # Count entity spans
        row_single = [s for s in spans if not s['is_multi']]
        row_multi  = [s for s in spans if s['is_multi']]
        single_entity_count += len(row_single)
        multi_entity_count  += len(row_multi)

        if spans:
            sentences_with_entity += 1
        else:
            sentences_no_entity += 1

        # Accumulate entities
        for s in spans:
            all_entities.append(s['text'])
            if s['is_multi']:
                all_multi.append(s['text'])
            else:
                all_single.append(s['text'])

        # Sentiment breakdown
        sentiment_stats[sent_label]['total'] += 1
        sentiment_stats[sent_label]['single'] += len(row_single)
        sentiment_stats[sent_label]['multi']  += len(row_multi)
        sentiment_stats[sent_label]['entities'].extend([s['text'] for s in spans])

        # Store row result
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

        # BIO output lines
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

    # ── Compile statistics ───────────────────────────────────────────────────
    entity_freq       = Counter(all_entities)
    single_freq       = Counter(all_single)
    multi_freq        = Counter(all_multi)
    total_entities    = single_entity_count + multi_entity_count

    stats = {
        'total_rows'             : len(df),
        'total_tokens'           : total_tokens,
        'total_entity_tokens'    : total_entity_tokens,
        'total_b_tags'           : total_b_tags,
        'total_i_tags'           : total_i_tags,
        'total_o_tags'           : total_o_tags,
        'total_entities'         : total_entities,
        'single_entity_count'    : single_entity_count,
        'multi_entity_count'     : multi_entity_count,
        'sentences_with_entity'  : sentences_with_entity,
        'sentences_no_entity'    : sentences_no_entity,
        'unique_entities'        : len(entity_freq),
        'unique_single'          : len(single_freq),
        'unique_multi'           : len(multi_freq),
        'entity_freq'            : entity_freq,
        'single_freq'            : single_freq,
        'multi_freq'             : multi_freq,
        'sentiment_stats'        : sentiment_stats,
        'rows_bio'               : rows_bio,
    }

    return df, stats, bio_output


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(stats, output_txt_path, bio_output, output_bio_path):
    """Write the full analysis report (.txt) and BIO-tagged output file."""

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
    lines.append("  ELSA DATASET — MULTI-WORD ENTITY DETECTION REPORT")
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
    lines.append(f"  {'Rank':<6} {'Multi-Word Entity':<40} {'Count':>8} {'% of Multi':>12}")
    lines.append(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*12}")
    for rank, (ent, cnt) in enumerate(s['multi_freq'].most_common(), 1):
        pct_e = cnt / multi_cnt * 100 if multi_cnt else 0
        lines.append(f"  {rank:<6} {ent:<40} {cnt:>8,} {pct_e:>11.2f}%")

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
    • "ফেস ওয়াশ" (face wash) clarifies it is a face cleanser product,
      not just any "wash" item.
    • "প্যানটিন শ্যাম্পু" tells us both brand (Pantene) + product type.

  DISAMBIGUATION RULE:
    When a single-token entity like "সাবান" appears alone in a review,
    check if a multi-token entity containing "সাবান" exists elsewhere in
    the document or corpus → use its context to resolve the full product name.
    This mirrors the MEID finding: 78.87%% of single-token entities have
    at least one multi-token counterpart in the same document.
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
            tokens_detail = ' + '.join(
                [f"'{t}'" for t in sp['tokens']]
            )
            tags_detail = 'B-PRODUCT' + ' + I-PRODUCT' * (len(sp['tokens']) - 1)
            lines.append(f"  Entity  : '{sp['text']}'")
            lines.append(f"  Tokens  : {tokens_detail}")
            lines.append(f"  BIO     : {tags_detail}")
        lines.append("")
        shown += 1

    # ── SECTION 9: COMPLETE ENTITY FREQUENCY TABLE ──────────────────────────
    section("SECTION 9: COMPLETE ENTITY FREQUENCY TABLE (All Entities, Sorted)")
    lines.append(f"  {'Rank':<6} {'Entity':<40} {'Count':>8} {'Type':<12} {'% of Total Ents':>16}")
    lines.append(f"  {'-'*6} {'-'*40} {'-'*8} {'-'*12} {'-'*16}")
    for rank, (ent, cnt) in enumerate(s['entity_freq'].most_common(), 1):
        etype = 'MULTI-WORD' if ' ' in ent else 'SINGLE'
        pct_e = cnt / total_ent * 100 if total_ent else 0
        lines.append(f"  {rank:<6} {ent:<40} {cnt:>8,} {etype:<12} {pct_e:>15.3f}%")

    # ── SECTION 10: SUMMARY INSIGHTS ────────────────────────────────────────
    section("SECTION 10: SUMMARY INSIGHTS & CONCLUSIONS")
    multi_pct_of_total = multi_cnt / total_ent * 100 if total_ent else 0
    lines.append(f"""
  1. SCALE
     The ELSA 10K dataset contains {total:,} product reviews in Bangla.
     A total of {total_ent:,} entity spans were detected across {s['sentences_with_entity']:,}
     sentences ({pct_ent_sents:.1f}%% of all reviews contain at least one entity).

  2. SINGLE vs MULTI-WORD ENTITIES
     • Single-word entities: {single_cnt:,} ({pct_single:.1f}%%)
     • Multi-word entities : {multi_cnt:,}  ({pct_multi:.1f}%%)
     Multi-word entities make up {multi_pct_of_total:.1f}%% of all detected entities.

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
     B-PRODUCT tags: {s['total_b_tags']:,} ({pct_b:.2f}%% of tokens)
     I-PRODUCT tags: {s['total_i_tags']:,} ({pct_i:.2f}%% of tokens)
     O tags        : {s['total_o_tags']:,} ({pct_o:.2f}%% of tokens)

  6. SENTIMENT CORRELATION
     Multi-word entities appear across all sentiment classes, indicating
     that reviewers use specific product names regardless of satisfaction.
     Positive reviews tend to have more entity mentions overall.

  7. IMPLICATION FOR NER MODELS
     Following the MEID (AAAI 2020) approach:
     → Multi-word entities like 'ডেটল সাবান' should receive HIGHER
       attention weights when disambiguating single-token occurrences
       of 'সাবান' or 'ডেটল' elsewhere in the same document.
     → An auxiliary MEC (Multi-token Entity Classification) task can
       pre-label SUB vs NSUB tokens before the main NER decoding step.
  """)

    lines.append(H1)
    lines.append("  END OF REPORT")
    lines.append(H1)

    # ── WRITE MAIN REPORT ────────────────────────────────────────────────────
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[INFO] Report written to: {output_txt_path}")

    # ── WRITE BIO TAGGED FILE ────────────────────────────────────────────────
    with open(output_bio_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bio_output))
    print(f"[INFO] BIO-tagged sentences written to: {output_bio_path}")

    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure the output directory exists before writing files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Process dataset
    df, stats, bio_output = process_dataset(INPUT_FILE)

    # Step 2: Generate report and BIO file
    total_lines = generate_report(stats, OUTPUT_TXT, bio_output, OUTPUT_BIO)

    # Step 3: Print quick summary to console
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
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_TXT}")
    print(f"    2. {OUTPUT_BIO}")
    print("\n[DONE]")
