import argparse
import csv
import json
from transformers import pipeline

def classify_texts(texts, labels, threshold=0.0):
    """
    Classify a list of texts against a set of candidate labels using
    HuggingFace's zero-shot-classification pipeline with a public model.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    all_results = []

    for text in texts:
        result = classifier(
            text,
            candidate_labels=labels,
            hypothesis_template="This text is about {}"
        )
        filtered = [
            {"label": label, "score": float(score)}
            for label, score in zip(result["labels"], result["scores"])
            if score >= threshold
        ]
        all_results.append({
            "text": text,
            "predictions": filtered
        })

    return all_results

def read_csv_column(path, column_name="text"):
    """
    Read a CSV file and return a list of values from the specified column.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row[column_name] for row in reader if row.get(column_name)]

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Topic Classifier (Public Model Only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Single text snippet to classify")
    group.add_argument("--file", "-f", type=str, help="Path to CSV file with a 'text' column")
    parser.add_argument(
        "--labels", "-l",
        type=lambda s: [label.strip() for label in s.split(",")],
        required=True,
        help="Comma-separated list of candidate labels"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum score to include a label in the output"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Optional path to save results as JSON"
    )

    args = parser.parse_args()

    if args.text:
        texts = [args.text]
    else:
        texts = read_csv_column(args.file)

    results = classify_texts(texts, args.labels, threshold=args.threshold)

    for r in results:
        print(f"\nText: {r['text']}")
        for pred in r["predictions"]:
            print(f"  {pred['label']:<15} {pred['score']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
