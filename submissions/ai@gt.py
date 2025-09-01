import numpy as np
import gradio as gr
from transformers import pipeline


# Pre-trained Sentiment Analysis
sent_pipe = pipeline( "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyse_sentiment(text):
    # Get prediction from the model
    result = sent_pipe(text)[0]
    label = result['label']
    confidence = result['score']
    return f"{label}", f"{confidence:.2%}",

# Create Gradio
demo = gr.Interface(
    fn=analyse_sentiment,inputs=gr.Textbox(
        lines=3, 
        placeholder="Enter a review, tweet, or any text to analyse...",
        label="Text to Analyse"
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence"),
    ],
)

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=True)

#public link to access: https://4fef37379d28bd6236.gradio.live/ 