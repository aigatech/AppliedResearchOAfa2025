import numpy as np
from PIL import Image
import gradio as gr

from main import (
    crop_person_center,
    lab_stats,
    average_hex,
    suggest_colors_with_hf_llm,
    save_named_palette_preview,
    guess_undertone,
)

def analyze(img: np.ndarray, status):
    if img is None:
        return None, None, "Please upload an image.", "❌ No image provided."

    status = "⏳ Processing... please wait."

    image = Image.fromarray(img).convert("RGB")

    face_region = crop_person_center(image)

    face_np = np.array(face_region)
    L, a, b, C, hue = lab_stats(face_np)
    skin_hex = average_hex(face_np)
    undertone = guess_undertone(a, b)

    pairs = suggest_colors_with_hf_llm(L, a, b, C, skin_hex, n=6)

    swatch_path = save_named_palette_preview(pairs, "palette_preview.png")

    lines = [
        f"**Sample HEX:** `{skin_hex}`",
        f"**Undertone (heuristic):** {undertone.upper()}",
        f"**CIELAB:** L* `{L:.1f}`, a* `{a:.1f}`, b* `{b:.1f}`, Chroma `{C:.1f}`, Hue `{hue:.0f}°`",
        "**HF-suggested colors:**",
    ]
    for name, hx in pairs:
        lines.append(f"- **{name}** — `{hx}`")
    explanation = "\n".join(lines)

    status = "Done!"
    return face_region, Image.open(swatch_path), explanation, status

with gr.Blocks() as demo:
    gr.Markdown("Skintone HF-Suggested Colors")
    gr.Markdown(
        "Detects the person (HF DETR), samples the central face region, computes CIELAB & average HEX, "
        "then asks a small Hugging Face text model (FLAN-T5) to recommend flattering colors. "
    )

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Upload a face photo")
        status_box = gr.Textbox(label="Status", value="Idle", interactive=False)

    analyze_btn = gr.Button("Analyze Image")

    with gr.Row():
        face_out = gr.Image(label="Analyzed face region")
        swatch_out = gr.Image(label="Suggested color swatches")
    details_out = gr.Markdown()

    analyze_btn.click(
        analyze,
        inputs=[img_input, status_box],
        outputs=[face_out, swatch_out, details_out, status_box]
    )

if __name__ == "__main__":
    demo.launch()
