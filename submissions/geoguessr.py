import os, math
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

# ---- Config knobs (env overrides) ----
HF_MODEL   = os.environ.get("HF_MODEL", "openai/clip-vit-base-patch32")
SPLIT      = os.environ.get("SPLIT", "train")   # "train" or "test" (if present)
N_SAMPLES  = int(os.environ.get("N_SAMPLES", "200"))  # how many to evaluate (streamed)
BATCH      = int(os.environ.get("BATCH", "8"))        # pipeline batch size
SEED       = int(os.environ.get("SEED", "42"))        # for streaming shuffle
BUFFER_SZ  = int(os.environ.get("BUFFER_SZ", "1000")) # shuffle buffer (streaming)

PROMPTS = [
    "a street scene in {country}",
    "a roadside photo in {country}",
    "a Google Street View photo in {country}",
    "an urban street in {country}",
    "a rural road in {country}",
]

def build_candidate_labels(labels: List[str]) -> Dict[str, List[str]]:
    return {c: [p.format(country=c) for p in PROMPTS] for c in labels}

def flatten(labels_by_country: Dict[str, List[str]]) -> List[str]:
    return [p for variants in labels_by_country.values() for p in variants]

def aggregate(raw_results, labels_by_country) -> List[Tuple[str, float]]:
    # map prompt -> country
    rev = {p: c for c, variants in labels_by_country.items() for p in variants}
    scores = {c: 0.0 for c in labels_by_country}
    for item in raw_results:
        c = rev.get(item["label"])
        if c is not None:
            scores[c] += float(item["score"])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main():
    print("Loading dataset in STREAMING mode (no full download)…")
    ds_stream = load_dataset("deboradum/GeoGuessr-countries", split=SPLIT, streaming=True)

    # In streaming, you can still access features for label names:
    labels = ds_stream.features["label"].names
    labels_by_country = build_candidate_labels(labels)
    candidate_labels_flat = flatten(labels_by_country)

    # Optional: shuffle the stream so you don’t just see the first files
    ds_stream = ds_stream.shuffle(seed=SEED, buffer_size=BUFFER_SZ)

    print(f"Model: {HF_MODEL}")
    clf = pipeline("zero-shot-image-classification", model=HF_MODEL)

    # Iterate lazily over N_SAMPLES, batching for speed
    total = 0
    top1 = 0
    top3 = 0
    samples = []

    batch_imgs, batch_truths = [], []
    pbar = tqdm(total=N_SAMPLES, desc="Evaluating (streamed)")

    for ex in ds_stream:
        img = ex["image"]          # PIL.Image
        gt  = labels[ex["label"]]  # country name string
        batch_imgs.append(img)
        batch_truths.append(gt)

        if len(batch_imgs) == BATCH:
            outs = clf(batch_imgs, candidate_labels=candidate_labels_flat)
            # outs is a list (per image) of prompt scores
            for raw, gt_country in zip(outs, batch_truths):
                ranked = aggregate(raw, labels_by_country)
                if ranked and ranked[0][0] == gt_country:
                    top1 += 1
                if any(c == gt_country for c, _ in ranked[:3]):
                    top3 += 1
                if len(samples) < 8:
                    samples.append({"gt": gt_country,
                                    "preds": [(c, round(s, 3)) for c, s in ranked[:3]]})
            total += len(batch_imgs)
            pbar.update(len(batch_imgs))
            batch_imgs, batch_truths = [], []

            if total >= N_SAMPLES:
                break

    # flush remaining
    if batch_imgs and total < N_SAMPLES:
        outs = clf(batch_imgs, candidate_labels=candidate_labels_flat)
        for raw, gt_country in zip(outs, batch_truths):
            ranked = aggregate(raw, labels_by_country)
            if ranked and ranked[0][0] == gt_country:
                top1 += 1
            if any(c == gt_country for c, _ in ranked[:3]):
                top3 += 1
            if len(samples) < 8:
                samples.append({"gt": gt_country,
                                "preds": [(c, round(s, 3)) for c, s in ranked[:3]]})
        total += len(batch_imgs)
        pbar.update(len(batch_imgs))

    pbar.close()
    top1_acc = top1 / total if total else 0.0
    top3_acc = top3 / total if total else 0.0

    print("\n===== Zero-shot streamed results =====")
    print(f"Split: {SPLIT} | Evaluated samples: {total} (streamed, no full download)")
    print(f"Top-1 accuracy: {top1_acc:.3f}")
    print(f"Top-3 accuracy: {top3_acc:.3f}")
    print("\nSample predictions:")
    for ex in samples:
        print(f"  GT: {ex['gt']:<18} Preds: {ex['preds']}")

    print("\nTips:")
    print("- Increase N_SAMPLES for a better estimate (still streamed).")
    print("- Try a larger CLIP model (HF_MODEL='openai/clip-vit-large-patch14') for quality (slower).")
    print("- Prompt engineering matters: tweak PROMPTS above and compare.")

if __name__ == "__main__":
    main()
