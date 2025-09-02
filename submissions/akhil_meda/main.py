from transformers import pipeline
import gradio as gr
import re

print("Loading the Models, Please wait. First time may take a couple minutes.\n")

#pipelines to huggingFace
toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert") #for toxicity
self_harm_classifier = pipeline("text-classification", model="vibhorag101/roberta-base-suicide-prediction-phr")#for suicide clasifer
rewriter = pipeline("text2text-generation", model="s-nlp/bart-base-detox")#to rewrite this into safer text

MAX_ATTEMPTS = 3  # fixed amt of attempts to rewrite the sentence

#Helper definistions 
def is_harmful(text: str) -> bool:
    toxic_result = toxic_classifier(text)[0]
    is_toxic = toxic_result['label'] == "toxic" and toxic_result['score'] > 0.7

    result = self_harm_classifier(text)[0]
    is_self_harm = result['label'] == 'Suicide post' and result['score'] > 0.8

    if is_toxic:
        print("Flagged by: General Toxicity Classifier")
        return True
    if is_self_harm:
        print("Flagged by: Self-Harm Classifier")
        return True
    
    return False

def hard_censor(text: str) -> str:
    return re.sub(r'\w+', lambda m: '*' * len(m.group(0)), text)

def rewrite_until_safe(message: str, max_attempts: int = MAX_ATTEMPTS) -> str:
    attempt = 0
    current = message

    while attempt < max_attempts and is_harmful(current):
        print(f"Harmful content detected (attempt {attempt+1}). Rewriting...")
        rewritten = rewriter(
            current,
            max_length=60
        )[0]["generated_text"]
        current = rewritten if rewritten.strip() else current
        attempt += 1

    if is_harmful(current):
        print("Cannot detoxify the message. Hard censor applied.")
        return hard_censor(current)
    
    return current

# Gradio definition for GUI
def analyze_and_rewrite(user_text: str):
    text = (user_text or "").strip()
    if not text:
        return "", "Neutral"

    initially_harmful = is_harmful(text)
    output = rewrite_until_safe(text, max_attempts=MAX_ATTEMPTS)
    final_harm = is_harmful(output)

    if final_harm:
        status = "Harmful"
    else:
        status = "Rewritten/Censored → Safe" if (initially_harmful and output != text) else "Safe"

    return output, status

with gr.Blocks(title="Harmful Comment Filter") as demo:
    gr.Markdown("## Harmful Comment Filter")
    inp = gr.Textbox(label="Your Message", lines=4, placeholder="Type anything...")
    out = gr.Textbox(label="Final Output", lines=4, interactive=False)
    status = gr.Label(label="Status")
    btn = gr.Button("Check & Rewrite")

    btn.click(analyze_and_rewrite, [inp], [out, status])
    inp.submit(analyze_and_rewrite, [inp], [out, status])

if __name__ == "__main__":
    demo.launch()