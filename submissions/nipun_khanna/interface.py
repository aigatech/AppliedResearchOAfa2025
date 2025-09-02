import gradio as gr
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

classifier = pipeline("sentiment-analysis", model=MODEL_PATH)

def analyze_sentiment(review):
    result = classifier(review)
    score = result[0]['score']
    return qualify_sentiment(score)

def qualify_sentiment(score):
    if score >= 0.9:
        return "Highly Positive"
    elif score >= 0.75:
        return "Positive"
    elif score >= 0.5:
        return "Neutral"
    elif score >= 0.25:
        return "Negative"
    else:
        return "Highly Negative"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Movie Review"),
    outputs=gr.Label(label="Sentiment"),
    title="Movie Review Sentiment Analyzer",
    description="Analyze the sentiment of movie reviews using machine learning!",
    flagging_mode="never",
    theme=gr.themes.Soft()
)

iface.launch(share=True)

