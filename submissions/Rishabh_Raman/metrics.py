"""
Quick metrics on predictions.csv: row count, label distribution, mean confidence.
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "predictions.csv"
df = pd.read_csv(path)

print("Rows:", len(df))
if "pred_label" not in df or "pred_score" not in df:
    raise SystemExit("predictions.csv missing required columns.")

print("\nLabel counts:")
print(df["pred_label"].value_counts())

print("\nMean confidence by label:")
print(df.groupby("pred_label")["pred_score"].mean().round(3).sort_values(ascending=False))
