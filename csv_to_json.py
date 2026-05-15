import argparse
import csv
import json
from pathlib import Path


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    with csv_path.open(mode="r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    with json_path.open(mode="w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=2)

    print(f"Converted {csv_path} -> {json_path} ({len(data)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSON.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="datasets/ELSA_Dataset - ELSA_10K.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        default="datasets/ELSA_Dataset - ELSA_10K.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    csv_to_json(input_path, output_path)
