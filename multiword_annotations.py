import json
import re


def clean_text(text):

    text = str(text)

    text = re.sub(r"[^\wঀ-৿\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
    "অসাধারণ",
    "গুড"
}


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


def valid_candidate(candidate):

    words = candidate.split()

    if len(words) < 2:
        return False

    if words[-1] in BAD_END_WORDS:
        return False

    for word in words:

        if word in GOOD_MODIFIERS:
            return True

    return False


def generate_multiword_entity(words, entity_index):

    entity = words[entity_index]

    total_words = len(words)

    candidates = []

    if entity_index >= 1:

        candidates.append(
            words[entity_index - 1]
            + " "
            + entity
        )

    if entity_index >= 2:

        candidates.append(
            words[entity_index - 2]
            + " "
            + words[entity_index - 1]
            + " "
            + entity
        )

    if (
        entity_index >= 1
        and entity_index < total_words - 1
    ):

        candidates.append(
            words[entity_index - 1]
            + " "
            + entity
            + " "
            + words[entity_index + 1]
        )

    best_candidate = entity

    meaningful = False

    for candidate in candidates:

        if valid_candidate(candidate):

            meaningful = True

            if len(candidate.split()) > len(
                best_candidate.split()
            ):

                best_candidate = candidate

    return best_candidate, meaningful


def generate_dataset(input_json, output_json):

    with open(input_json, "r", encoding="utf-8") as f:

        data = json.load(f)

    new_dataset = []

    for row in data:

        review = clean_text(
            row.get("REVIEW", "")
        )

        tagged_review = str(
            row.get("ENTITY_TAGGED_REVIEW", "")
        )

        sentiment = row.get(
            "ENTITY_SENTIMENT",
            ""
        )

        words = review.split()

        multiword_entity = ""

        meaningful = False

        match = re.search(
            r"_NE_([^\s]+)",
            tagged_review
        )

        if match:

            entity = clean_text(
                match.group(1)
            )

            for i, word in enumerate(words):

                if word == entity:

                    (
                        multiword_entity,
                        meaningful
                    ) = generate_multiword_entity(
                        words,
                        i
                    )

                    break

        new_row = {
            "REVIEW": review,
            "SINGLE_WORD_ENTITY": tagged_review,
            "MULTIWORD_ENTITY": multiword_entity,
            "MEANINGFUL_MULTIWORD": meaningful,
            "ENTITY_SENTIMENT": sentiment
        }

        new_dataset.append(new_row)

    with open(output_json, "w", encoding="utf-8") as f:

        json.dump(
            new_dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nDataset saved:\n"
        f"{output_json}"
    )


if __name__ == "__main__":

    generate_dataset(
        "datasets/ELSA_Dataset - ELSA_10K.json",
        "datasets/ELSA_10K_advanced_multiword.json"
    )