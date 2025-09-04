# Imports
import re
from typing import List
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr


# Configuration
# Model ID and generation parameters

MODEL_ID = "Salesforce/blip-image-captioning-base"

GEN_KW = dict(
    max_length=32,
    do_sample=False,
    top_p=0.90,
    temperature=0.85,
    num_return_sequences=1,
    no_repeat_ngram_size=3,
    early_stopping=True,
)

# Object list for caption-referencin.
GRABBABLE_PHRASES = {
    "coffee cup", "paper cup", "water bottle", "desk lamp", "computer mouse",
    "wireless mouse", "keyboard tray", "sticky notes", "paper clip",
    "notebook computer", "laptop computer", "clip board", "phone charger",
}
GRABBABLE_SINGLE = {
    "desk", "monitor", "screen", "keyboard", "mouse", "trackpad", "laptop",
    "tablet", "phone", "calculator", "clipboard", "notebook", "paper",
    "mug", "cup", "bottle", "plant", "chair", "pen", "pencil", "marker",
    "stapler", "tape", "scissors", "ruler", "headphones", "charger", "cable",
    "computer", "clock", "lamp", "picture", "book", "speakers",
}


# Model loading
# Use the default processor (fast tokenizer if available) and load the model

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)


# Helper functions for captions and lists

def _normalize(text: str) -> str:
    t = text.lower()
    for ch in '.,:;!?\"\'()[]{}':
        t = t.replace(ch, "")
    return t.strip()

def _format_list(items: List[str]) -> str:
    if not items:
        return "none detected"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# Object extraction from caption

def extract_grabbables(caption: str) -> List[str]:
    norm = _normalize(caption)

    phrases = []
    covered = set()
    for phrase in sorted(GRABBABLE_PHRASES):
        if re.search(r"\b" + re.escape(phrase) + r"\b", norm):
            phrases.append(phrase)
            covered.update(phrase.split())

    singles = sorted({w for w in norm.split() if w in GRABBABLE_SINGLE and w not in covered})
    return phrases + singles


# Caption generation functions

def _prepare_inputs(image: Image.Image):
    return processor(images=image, return_tensors="pt")

def generate_caption(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = _prepare_inputs(image)
    with torch.no_grad():
        output_ids = model.generate(**inputs, **GEN_KW)

    caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    low = caption.lower()
    for pfx in ("a photo of ", "an image of ", "a picture of "):
        if low.startswith(pfx):
            caption = caption[len(pfx):].strip()
            break
    return caption


# Produce caption and object highlights

def caption_and_targets(image: Image.Image):
    if image is None:
        return "Please upload an image to caption.", []

    caption = generate_caption(image)
    objs = extract_grabbables(caption)

    final = caption
    if final and not final.endswith("."):
        final += "."
    final += f" Objects: {_format_list(objs)}."

    highlights = [(o, "target") for o in objs]
    return final, highlights


# UI with Gradio

with gr.Blocks(title="Simple Robot Vision Captioning (Office Desk Only)") as demo:
    gr.Markdown(
        "# Simple Robot Vision Captioning (Office Desk Only)\n"
        "Upload an office-desk image. The app generates one caption and lists "
        "**grabbable office items** found in that caption."
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload office-desk image")
            btn = gr.Button("Caption Image")
        with gr.Column():
            caption_tb = gr.Textbox(label="Caption", interactive=False)
            targets_view = gr.HighlightedText(
                label="Grabbable objects (from caption)", combine_adjacent=True
            )

    btn.click(fn=caption_and_targets, inputs=[img_input], outputs=[caption_tb, targets_view])


# Entrypoint

if __name__ == "__main__":
    demo.launch(show_error=True)
