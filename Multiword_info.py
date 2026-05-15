import json
import re
from collections import Counter
from pathlib import Path


def generate_multiword_annotation_report(json_path: Path, output_txt: Path):

    # -----------------------------------
    # LOAD JSON DATA
    # -----------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)

    # -----------------------------------
    # VARIABLES
    # -----------------------------------
    total_entity_annotations = 0

    entity_counter = Counter()

    sentiment_counter = Counter()

    review_lengths = []

    candidate_multiword_phrases = []

    candidate_counter = Counter()

    # Bengali + English words
    phrase_pattern = re.compile(r'[\wঀ-৿]+', re.UNICODE)

    # -----------------------------------
    # PROCESS DATA
    # -----------------------------------
    for row in data:

        review = str(row.get("REVIEW", "")).strip()

        tagged_review = str(row.get("ENTITY_TAGGED_REVIEW", "")).strip()

        sentiment = str(row.get("ENTITY_SENTIMENT", "")).strip()

        review_lengths.append(len(review.split()))

        # -----------------------------------
        # FIND SINGLE TOKEN ENTITY
        # -----------------------------------

        entity_match = re.search(r'_NE_([^\s]+)', tagged_review)

        if entity_match:

            entity = entity_match.group(1).strip()

            total_entity_annotations += 1

            entity_counter[entity] += 1

            sentiment_counter[sentiment] += 1

            # -----------------------------------
            # MULTIWORD CANDIDATE EXTRACTION
            # -----------------------------------

            # find entity position in review
            if entity in review:

                words = review.split()

                for i, word in enumerate(words):

                    if entity in word:

                        # previous + entity
                        if i > 0:
                            phrase = words[i - 1] + " " + words[i]
                            candidate_multiword_phrases.append(phrase)
                            candidate_counter[phrase] += 1

                        # entity + next
                        if i < len(words) - 1:
                            phrase = words[i] + " " + words[i + 1]
                            candidate_multiword_phrases.append(phrase)
                            candidate_counter[phrase] += 1

                        # previous + entity + next
                        if i > 0 and i < len(words) - 1:
                            phrase = (
                                words[i - 1]
                                + " "
                                + words[i]
                                + " "
                                + words[i + 1]
                            )

                            candidate_multiword_phrases.append(phrase)
                            candidate_counter[phrase] += 1

    # -----------------------------------
    # REPORT
    # -----------------------------------

    report = []

    report.append("========== ELSA DATASET ANALYSIS ==========\n")

    report.append(f"Total Records: {total_records}")
    report.append(f"Total Entity Annotations: {total_entity_annotations}")
    report.append(
        f"Unique Single Token Entities: {len(entity_counter)}"
    )

    report.append(
        f"Generated Multiword Candidates: {len(candidate_multiword_phrases)}"
    )

    report.append(
        f"Unique Multiword Candidates: {len(candidate_counter)}\n"
    )

    # -----------------------------------
    # SENTIMENT
    # -----------------------------------

    report.append("========== SENTIMENT DISTRIBUTION ==========\n")

    report.append(f"Negative (0): {sentiment_counter['0']}")
    report.append(f"Neutral  (1): {sentiment_counter['1']}")
    report.append(f"Positive (2): {sentiment_counter['2']}\n")

    # -----------------------------------
    # REVIEW STATS
    # -----------------------------------

    avg_review_len = sum(review_lengths) / len(review_lengths)

    report.append("========== REVIEW STATISTICS ==========\n")

    report.append(f"Average Review Length: {avg_review_len:.2f} words")
    report.append(
        f"Maximum Review Length: {max(review_lengths)} words"
    )
    report.append(
        f"Minimum Review Length: {min(review_lengths)} words\n"
    )

    # -----------------------------------
    # TOP SINGLE ENTITIES
    # -----------------------------------

    report.append("========== TOP SINGLE TOKEN ENTITIES ==========\n")

    for entity, count in entity_counter.most_common(50):
        report.append(f"{entity} -> {count}")

    # -----------------------------------
    # TOP MULTIWORD CANDIDATES
    # -----------------------------------

    report.append("\n========== TOP MULTIWORD CANDIDATES ==========\n")

    for phrase, count in candidate_counter.most_common(100):

        # remove very short/noisy phrases
        if len(phrase.split()) >= 2:

            report.append(f"{phrase} -> {count}")

    # -----------------------------------
    # SAVE TXT REPORT
    # -----------------------------------

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to:\n{output_txt}")


if __name__ == "__main__":

    json_file = Path(
        "datasets/ELSA_Dataset - ELSA_10K.json"
    )

    output_file = Path(
        "datasets/multiword_annotation_analysis.txt"
    )

    generate_multiword_annotation_report(
        json_file,
        output_file
    )