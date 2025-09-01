from transformers import pipeline
import gradio as gr
from datetime import datetime

# load emotion model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# store history
history = []

def analyze_text(text):
    if not text:
        return "please enter some text"
    
    result = classifier(text)
    emotion = result[0]['label']
    score = result[0]['score']
    
    # add to history
    history.append({
        'time': datetime.now().strftime("%H:%M"),
        'text': text[:40] + "..." if len(text) > 40 else text,
        'emotion': emotion
    })
    
    return f"Emotion: {emotion} ({score:.1%} confident)"

def analyze_batch(texts):
    if not texts:
        return "enter some text please"
    
    lines = texts.split('\n')
    results = []
    
    for line in lines:
        if line.strip():
            result = classifier(line.strip())
            emotion = result[0]['label']
            short_text = line[:25] + "..." if len(line) > 25 else line
            results.append(f"{short_text} -> {emotion}")
    
    return '\n'.join(results)

def show_history():
    if not history:
        return "no history yet"
    
    output = ""
    for item in history:
        output += f"{item['time']}: {item['text']} -> {item['emotion']}\n"
    return output

def clear_hist():
    global history
    history = []
    return "cleared!"

# make the interface
with gr.Blocks() as app:
    gr.Markdown("# Emotion Detector")
    gr.Markdown("detects emotions in text using AI")
    
    with gr.Tab("Single Text"):
        text_input = gr.Textbox(label="enter text", placeholder="type something here")
        text_output = gr.Textbox(label="result")
        analyze_btn = gr.Button("analyze")
        
    with gr.Tab("Multiple Texts"):
        batch_input = gr.Textbox(label="enter texts (one per line)", lines=4)
        batch_output = gr.Textbox(label="results", lines=8)
        batch_btn = gr.Button("analyze all")
    
    with gr.Tab("History"):
        hist_output = gr.Textbox(label="history", lines=10)
        show_btn = gr.Button("show history")
        clear_btn = gr.Button("clear")
    
    # connect buttons
    analyze_btn.click(analyze_text, inputs=text_input, outputs=text_output)
    batch_btn.click(analyze_batch, inputs=batch_input, outputs=batch_output)
    show_btn.click(show_history, outputs=hist_output)
    clear_btn.click(clear_hist, outputs=hist_output)

if __name__ == "__main__":
    app.launch()