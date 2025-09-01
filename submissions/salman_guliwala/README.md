# BLIP Image Captioner & Sentiment Analyzer

## What it does

This application uses AI to automatically caption uploaded images and analyze the emotional sentiment of those captions. It combines computer vision (BLIP model) with natural language processing (sentiment analysis) to provide both descriptive and emotional insights about images.

## How to run it

1. Install dependencies:
   `pip install transformers torch torchvision gradio pillow`

2. Run the application:
   `python image_captioner.py`

3. Open the provided URL in your browser and start uploading images!

## Features

- ✨ Automatic image captioning using BLIP model
- 🎭 Sentiment analysis of generated captions
- 🌐 Easy-to-use web interface
- 😊 Emoji-enhanced results
- ⚡ Runs on CPU (no GPU required)

## Models Used

- `Salesforce/blip-image-captioning-base` for image captioning
- `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment analysis
