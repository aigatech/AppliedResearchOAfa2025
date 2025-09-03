import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import random
import time

# Load model once
model_id = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_id)
model = M2M100ForConditionalGeneration.from_pretrained(model_id)

# Remove EN since we only use it to start and end
LANGUAGES = [l for l in list(tokenizer.lang_code_to_id.keys()) if l != "en"]

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_new_tokens=60
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def lost_in_translation_stream(text, steps=5):
    current_lang = "en"
    current_text = text
    history = [f"Original (en): {text}"]

    for i in range(steps):
        next_lang = random.choice([l for l in LANGUAGES if l != current_lang])
        translated = translate(current_text, current_lang, next_lang)
        history.append(f"{current_lang} → {next_lang}: {translated}")
        current_text, current_lang = translated, next_lang
        # Yield the updated history at each step
        yield "\n".join(history)

    final = translate(current_text, current_lang, "en")
    history.append(f"{current_lang} → en: {final}")
    yield "\n".join(history)

# Gradio Interface using streaming
demo = gr.Interface(
    fn=lost_in_translation_stream,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a sentence to get lost in translation"),
        gr.Slider(1, 10, value=5, step=1, label="Number of translation hops")
    ],
    outputs=gr.Textbox(lines=15),
    title="🌐 Lost in Translation Game",
    description="Watch your sentence change after each translation hop!"
)

if __name__ == "__main__":
    demo.launch()
