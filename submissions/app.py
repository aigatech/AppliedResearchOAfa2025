import re
import gradio as gr
from transformers import pipeline

GEN = pipeline("text2text-generation", model="google/flan-t5-small")

def clamp(x, a=0, b=100): 
    return max(a, min(b, int(x)))

def style_from_sophistication(level: int):
    """Return audience/instructions and a *base* word target that grows with sophistication."""
    level = clamp(level)
    if level < 34:
        return {
            "audience": "a curious 10-year-old",
            "instruction": "Explain in very simple language without jargon.",
            "target_words_base": 120,   #short
            "sentences": "4–6"
        }
    elif level < 67:
        return {
            "audience": "a college freshman",
            "instruction": "Use accessible terms and define any jargon.",
            "target_words_base": 280,   #medium
            "sentences": "6–9"
        }
    else:
        return {
            "audience": "a domain expert",
            "instruction": "Use technical language and include context, caveats, and implications.",
            "target_words_base": 400,   #longer
            "sentences": "8–12"
        }

def summarize(text: str, sophistication: int) -> str:
    if not text.strip():
        return "Please paste text to summarize."

    style = style_from_sophistication(sophistication)
    src_words = len(text.split())

    # Cap target by source length
    # Allows ~120 words beyond 90% of source size
    target_words = min(style["target_words_base"], int(src_words * 0.9 + 120))

    # Rough word to token budget
    max_toks = max(120, int(target_words * 1.6))
    min_toks = max(80, int(max_toks * 0.6))

    prompt = (
        f"Summarize for {style['audience']}. {style['instruction']}\n"
        f"Write {style['sentences']} sentences in a single paragraph. No bullets. "
        "Avoid repeating any sentence or phrase. Keep facts faithful to the text; do not invent details.\n\n"
        f"TEXT:\n{text}"
    )

    out = GEN(
        prompt,
        max_new_tokens=max_toks,
        min_new_tokens=min_toks,       # enforce longer outputs when sophistication is high
        do_sample=True,
        temperature=0.75,
        top_p=0.9,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        truncation=True,
    )
    return out[0]["generated_text"]

###If you are reading this, first of all, thank you for your time! More importantly, I personally was hoping to get a true gradient
###of expertise rather than discrete categories, but out of respect for the time constraints I decided to stop at this proof of concept.
###I know this may not be the most creative submission you see, but it's something that I will continue to build out and hopefully use regularly!

with gr.Blocks(title="Simple Adjustable Text Summarizer") as demo:
    gr.Markdown("# Simple Adjustable Text Summarizer\nInspired by my constant battles of asking LLM's to summarize a paper, but give me more detail, no less detail, just a little more....")
    gr.Markdown("Notes:\nThere is a slight UI bug in that the first time you click the summarize button, the loading effect sometimes doesn't trigger. Each summary should take about 30 seconds (tested with up to 10,000 characters).")
    text = gr.Textbox(label="Text", lines=10, placeholder="Paste an abstract or section…")
    sophistication = gr.Slider(0, 100, value=50, step=1, label="Sophistication")
    run_btn = gr.Button("Summarize")
    out = gr.Markdown(label="Summary")
    run_btn.click(summarize, inputs=[text, sophistication], outputs=out)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
