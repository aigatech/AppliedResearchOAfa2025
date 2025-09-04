# file: zs_demo.py
import gradio as gr
from transformers import pipeline
from datasets import load_dataset

clf = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
labels = load_dataset("deboradum/GeoGuessr-countries", split="train", streaming=True).features["label"].names
PROMPTS = ["a street scene in {country}", "a Google Street View photo in {country}"]
candidates = [p.format(country=c) for c in labels for p in PROMPTS]

def predict(img):
    preds = clf(img, candidate_labels=candidates)
    # aggregate per country
    scores = {}
    for p in preds:
        country = p["label"].split(" in ", 1)[1]
        scores[country] = scores.get(country, 0) + float(p["score"])
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return {k: float(v) for k, v in top3}

gr.Interface(predict, gr.Image(type="pil"), gr.Label(num_top_classes=3),
             title="Zero-Shot GeoGuessr (toy)").launch()
