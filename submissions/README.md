# AI@GT Applied Research Fall 2025 - Image Story Generator

## What it does

This app takes an image and creates a children's story from it. It works in two steps:
1. Generates a caption describing what's in the image
2. Expands that caption into a short children's story

## How it works

The app has three main parts:
- captioner.py: Creates captions from images using a BLIP model
- expand.py: Turns captions into stories using Google's Gemma model  
- frontend.py: Web interface built with Streamlit

## How to use

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run frontend.py
   ```

3. Open your browser to the URL shown in the terminal

4. Upload an image and click "Generate Caption" to create a story

## Requirements

- Python 3.7 or higher
- need internet
- The packages listed in requirements.txt
- HuggingFace API token (get one from https://huggingface.co/settings/tokens)

## Setup

1. Copy `env_template.txt` to `.env`
2. Add your HuggingFace API token to the `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Example

Upload a photo of a cat → Get caption "A cat playing in a garden" → Get story "Once upon a time..."

---

