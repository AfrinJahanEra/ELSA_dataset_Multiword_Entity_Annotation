import json
import re
from collections import Counter


def clean_text(text):

    text = str(text)

    text = re.sub(r"[^\wঀ-৿\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


BAD_END_WORDS = {
    "পেয়েছি",
    "ধন্যবাদ",
    "এবং",
    "সবাই",
    "ছিল",
    "হয়",
    "হলো",
    "করেছে",
    "করলাম",
    "চাইলে",
    "আছে"
}


GOOD_MODIFIERS = {
    "ভালো",
    "ভাল",
    "খুব",
    "অনেক",
    "সুন্দর",
    "অরিজিনাল",
    "অরজিনাল",
    "সেরা",
    "একটি",
    "একটা",
    "দামি",
    "মানের",
    "চমৎকার",
    "অসাধারণ"
}


def generate_phrases(words, entity_index):

    phrases = []

    total_words = len(words)

    entity = words[entity_index]

    if entity_index >= 1:

        phrases.append(
            words[entity_index - 1]
            + " "
            + entity
        )

    if entity_index < total_words - 1:

        phrases.append(
            entity
            + " "
            + words[entity_index + 1]
        )

    if entity_index >= 2:

        phrases.append(
            words[entity_index - 2]
            + " "
            + words[entity_index - 1]
            + " "
            + entity
        )

    if entity_index >= 1 and entity_index < total_words - 1:

        phrases.append(
            words[entity_index - 1]
            + " "
            + entity
            + " "
            + words[entity_index + 1]
        )

    if entity_index < total_words - 2:

        phrases.append(
            entity
            + " "
            + words[entity_index + 1]
            + " "
            + words[entity_index + 2]
        )

    return phrases


def valid_phrase(phrase):

    words = phrase.split()

    if len(words) < 2:
        return False

    if words[-1] in BAD_END_WORDS:
        return False

    for word in words:

        if word in GOOD_MODIFIERS:
            return True

    return False


def analyze_dataset(input_json, output_report):

    with open(input_json, "r", encoding="utf-8") as f:

        data = json.load(f)

    total_records = len(data)

    total_entities = 0

    total_multiword_generated = 0

    two_word_count = 0

    three_word_count = 0

    entity_counter = Counter()

    phrase_counter = Counter()

    sentiment_counter = Counter()

    review_lengths = []

    unique_reviews = set()

    for row in data:

        review = clean_text(
            row.get("REVIEW", "")
        )

        tagged_review = str(
            row.get("ENTITY_TAGGED_REVIEW", "")
        )

        sentiment = str(
            row.get("ENTITY_SENTIMENT", "")
        )

        unique_reviews.add(review)

        sentiment_counter[sentiment] += 1

        words = review.split()

        review_lengths.append(len(words))

        match = re.search(
            r"_NE_([^\s]+)",
            tagged_review
        )

        if not match:
            continue

        entity = clean_text(
            match.group(1)
        )

        total_entities += 1

        entity_counter[entity] += 1

        for i, word in enumerate(words):

            if word == entity:

                phrases = generate_phrases(
                    words,
                    i
                )

                for phrase in phrases:

                    if valid_phrase(phrase):

                        phrase_counter[phrase] += 1

                        total_multiword_generated += 1

                        if len(phrase.split()) == 2:

                            two_word_count += 1

                        elif len(phrase.split()) == 3:

                            three_word_count += 1

                break

    avg_review_length = (
        sum(review_lengths)
        / len(review_lengths)
    )

    max_review_length = max(review_lengths)

    min_review_length = min(review_lengths)

    unique_multiword = len(phrase_counter)

    unique_single_entities = len(entity_counter)

    positive_percent = (
        sentiment_counter["2"]
        / total_records
    ) * 100

    neutral_percent = (
        sentiment_counter["1"]
        / total_records
    ) * 100

    negative_percent = (
        sentiment_counter["0"]
        / total_records
    ) * 100

    two_word_percent = (
        two_word_count
        / total_multiword_generated
    ) * 100

    three_word_percent = (
        three_word_count
        / total_multiword_generated
    ) * 100

    entity_coverage_percent = (
        total_entities
        / total_records
    ) * 100

    report = []

    report.append(
        "========== ADVANCED MULTIWORD DATASET ANALYSIS ==========\n"
    )

    report.append(
        "========== DATASET OVERVIEW ==========\n"
    )

    report.append(
        f"Total Records: {total_records}"
    )

    report.append(
        f"Unique Reviews: {len(unique_reviews)}"
    )

    report.append(
        f"Total Entity Annotations: {total_entities}"
    )

    report.append(
        f"Entity Coverage: "
        f"{entity_coverage_percent:.2f}%"
    )

    report.append(
        f"Unique Single Entities: "
        f"{unique_single_entities}"
    )

    report.append(
        f"Generated Multiword Candidates: "
        f"{total_multiword_generated}"
    )

    report.append(
        f"Unique Multiword Candidates: "
        f"{unique_multiword}\n"
    )

    report.append(
        "========== SENTIMENT DISTRIBUTION ==========\n"
    )

    report.append(
        f"Positive (2): "
        f"{sentiment_counter['2']} "
        f"({positive_percent:.2f}%)"
    )

    report.append(
        f"Neutral (1): "
        f"{sentiment_counter['1']} "
        f"({neutral_percent:.2f}%)"
    )

    report.append(
        f"Negative (0): "
        f"{sentiment_counter['0']} "
        f"({negative_percent:.2f}%)\n"
    )

    report.append(
        "========== REVIEW STATISTICS ==========\n"
    )

    report.append(
        f"Average Review Length: "
        f"{avg_review_length:.2f} words"
    )

    report.append(
        f"Maximum Review Length: "
        f"{max_review_length} words"
    )

    report.append(
        f"Minimum Review Length: "
        f"{min_review_length} words\n"
    )

    report.append(
        "========== MULTIWORD STATISTICS ==========\n"
    )

    report.append(
        f"2 Word Candidates: "
        f"{two_word_count} "
        f"({two_word_percent:.2f}%)"
    )

    report.append(
        f"3 Word Candidates: "
        f"{three_word_count} "
        f"({three_word_percent:.2f}%)\n"
    )

    report.append(
        "========== TOP SINGLE TOKEN ENTITIES ==========\n"
    )

    for entity, count in entity_counter.most_common(100):

        percent = (
            count / total_entities
        ) * 100

        report.append(
            f"{entity} -> "
            f"{count} "
            f"({percent:.2f}%)"
        )

    report.append(
        "\n========== TOP MULTIWORD CANDIDATES ==========\n"
    )

    for phrase, count in phrase_counter.most_common(200):

        percent = (
            count / total_multiword_generated
        ) * 100

        report.append(
            f"{phrase} -> "
            f"{count} "
            f"({percent:.2f}%)"
        )

    with open(output_report, "w", encoding="utf-8") as f:

        f.write("\n".join(report))

    print(
        f"\nDetailed analysis saved:\n"
        f"{output_report}"
    )


if __name__ == "__main__":

    analyze_dataset(
        "datasets/ELSA_Dataset - ELSA_10K.json",
        "datasets/detailed_multiword_analysis.txt"
    )