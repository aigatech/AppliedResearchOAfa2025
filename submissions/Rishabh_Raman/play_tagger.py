"""
NFL Play Tagger (Zero-Shot)
Author: Rishabh Raman

Classifies short NFL play descriptions into one of:
  touchdown, field goal, turnover, penalty, pass, rush, timeout, injury
"""

import argparse
import pandas as pd
from transformers import pipeline

# NFL-specific labels (kept small on purpose)
NFL_LABELS = [
    "touchdown",
    "field goal",
    "turnover",
    "penalty",
    "pass",
    "rush",
    "timeout",
    "injury",
]

def build_classifier(model_name: str = "typeform/distilbert-base-uncased-mnli"):
    """Create zero-shot pipeline on CPU."""
    return pipeline("zero-shot-classification", model=model_name, device=-1)

def classify_texts(clf, texts, labels, multi_label=False):
    """Run zero-shot classification and normalize to a list."""
    out = clf(texts, candidate_labels=labels, multi_label=multi_label)
    return out if isinstance(out, list) else [out]

def heuristic_hint(text: str):
    """
    Tiny domain hint to show awareness (doesn't override predictions).
    """
    t = text.lower()
    if "sack" in t:
        return "Hint: contains 'sack' → negative play; turnover only if fumble/lost ball."
    if "extra point" in t:
        return "Hint: extra point attempt → usually follows a touchdown."
    return None

def run_single(clf, text, labels, multi_label):
    res = classify_texts(clf, [text], labels, multi_label)[0]
    print(f"\nText: {text}")
    print(f"Top-1: {res['labels'][0]} ({res['scores'][0]:.3f})")
    top3 = list(zip(res["labels"][:3], [round(s, 3) for s in res["scores"][:3]]))
    print("Top-3:", top3)
    hint = heuristic_hint(text)
    if hint:
        print(hint)

def run_csv(clf, in_csv, out_csv, text_col, labels, multi_label):
    df = pd.read_csv(in_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {in_csv}. Columns: {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{in_csv} is empty.")
    preds = classify_texts(clf, df[text_col].tolist(), labels, multi_label)
    df["pred_label"] = [p["labels"][0] for p in preds]
    df["pred_score"] = [float(p["scores"][0]) for p in preds]
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="NFL Play Tagger (Zero-Shot)")
    ap.add_argument("--model", default="typeform/distilbert-base-uncased-mnli")
    ap.add_argument("--labels", nargs="*", default=NFL_LABELS)
    ap.add_argument("--multi_label", action="store_true")
    ap.add_argument("--mode", choices=["single", "csv"], required=True)
    ap.add_argument("--text", help="Play text (single mode)")
    ap.add_argument("--in_csv", help="Input CSV path (csv mode)")
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--out_csv", default="predictions.csv")
    args = ap.parse_args()

    clf = build_classifier(args.model)

    if args.mode == "single":
        if not args.text:
            raise SystemExit("Provide --text for single mode")
        run_single(clf, args.text, args.labels, args.multi_label)
    else:
        if not args.in_csv:
            raise SystemExit("Provide --in_csv for csv mode")
        run_csv(clf, args.in_csv, args.out_csv, args.text_col, args.multi_label)

if __name__ == "__main__":
    main()
