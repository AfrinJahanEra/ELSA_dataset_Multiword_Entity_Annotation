import json
import re
from pathlib import Path


def create_multiword_annotations(input_json, output_json):

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = []

    # -----------------------------------
    # PROCESS EACH ROW
    # -----------------------------------

    for row in data:

        review = str(row.get("REVIEW", "")).strip()

        tagged_review = str(
            row.get("ENTITY_TAGGED_REVIEW", "")
        ).strip()

        sentiment = row.get("ENTITY_SENTIMENT", "")

        # -----------------------------------
        # FIND ENTITY
        # -----------------------------------

        entity_match = re.search(
            r'_NE_([^\s]+)',
            tagged_review
        )

        multiword_entity = ""

        if entity_match:

            entity = entity_match.group(1).strip()

            words = review.split()

            # -----------------------------------
            # FIND ENTITY POSITION
            # -----------------------------------

            for i, word in enumerate(words):

                if entity in word:

                    # -----------------------------------
                    # BUILD MULTIWORD ENTITY
                    # -----------------------------------

                    left_word = ""
                    right_word = ""

                    if i > 0:
                        left_word = words[i - 1]

                    if i < len(words) - 1:
                        right_word = words[i + 1]

                    # -----------------------------------
                    # CREATE CANDIDATES
                    # -----------------------------------

                    candidates = []

                    if left_word:
                        candidates.append(
                            left_word + " " + words[i]
                        )

                    if right_word:
                        candidates.append(
                            words[i] + " " + right_word
                        )

                    if left_word and right_word:
                        candidates.append(
                            left_word
                            + " "
                            + words[i]
                            + " "
                            + right_word
                        )

                    # -----------------------------------
                    # CHOOSE LONGEST CANDIDATE
                    # -----------------------------------

                    if candidates:
                        multiword_entity = max(
                            candidates,
                            key=len
                        )

                    break

        # -----------------------------------
        # ADD NEW FIELD
        # -----------------------------------

        new_row = {
            "REVIEW": review,
            "SINGLE_WORD_ENTITY": tagged_review,
            "MULTIWORD_ANNOTATION": multiword_entity,
            "ENTITY_SENTIMENT": sentiment
        }

        updated_data.append(new_row)

    # -----------------------------------
    # SAVE NEW JSON
    # -----------------------------------

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            updated_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nMultiword annotation dataset saved:\n{output_json}"
    )


if __name__ == "__main__":

    input_file = (
        "datasets/ELSA_Dataset - ELSA_10K.json"
    )

    output_file = (
        "datasets/ELSA_10K_multiword.json"
    )

    create_multiword_annotations(
        input_file,
        output_file
    )