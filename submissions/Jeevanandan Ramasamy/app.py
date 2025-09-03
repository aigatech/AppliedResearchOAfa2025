import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import random

# Load model once
model_id = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_id)
model = M2M100ForConditionalGeneration.from_pretrained(model_id)

# Popular languages (short-list)
POPULAR_LANGUAGES = ["fr", "de", "es", "ru", "ja", "it", "nl", "zh", "ar", "hi", "pt", "ko"]

# All supported languages except English
ALL_LANGUAGES = [l for l in list(tokenizer.lang_code_to_id.keys()) if l != "en"]

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_new_tokens=60
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def lost_in_translation_stream(text, steps=5, use_all_languages=False):
    current_lang = "en"
    current_text = text
    history = [f"Original (en): {text}"]
    used_languages = set()

    LANGUAGES = ALL_LANGUAGES if use_all_languages else POPULAR_LANGUAGES

    for _ in range(steps):
        next_lang_options = [l for l in LANGUAGES if l != current_lang and l not in used_languages]
        if not next_lang_options:
            # Reset used languages if exhausted
            used_languages.clear()
            next_lang_options = [l for l in LANGUAGES if l != current_lang]

        next_lang = random.choice(next_lang_options)
        used_languages.add(next_lang)

        translated = translate(current_text, current_lang, next_lang)
        history.append(f"{current_lang} → {next_lang}: {translated}")
        current_text, current_lang = translated, next_lang
        yield "\n".join(history)

    final_translation = translate(current_text, current_lang, "en")
    history.append(f"{current_lang} → en: {final_translation}")
    yield "\n".join(history)

# Gradio interface
demo = gr.Interface(
    fn=lost_in_translation_stream,
    inputs=[
        gr.Textbox(lines=3, label="Input", placeholder="Type your sentence here..."),
        gr.Slider(1, 10, value=5, step=1, label="Number of translation hops"),
        gr.Checkbox(label="Use all supported languages", value=False),
        gr.Markdown("⚠️ Using 'all supported languages' may produce weird or less accurate translations.")
    ],
    outputs=gr.Textbox(lines=15, label="Translation History", placeholder="Translation steps will appear here..."),
    title="🌐 Lost in Translation Game",
    description=(
        "Watch your sentence travel the world, one language at a time! 🌎\n\n"
        "🎯 How it works:\n"
        "- Your sentence starts in English.\n"
        "- It gets translated randomly across multiple languages.\n"
        "- Finally, it comes back to English… often hilariously different!\n\n"
        "💡 Fun tip: Try with idioms or tongue twisters for maximum chaos!"
    )
)

if __name__ == "__main__":
    demo.launch()
