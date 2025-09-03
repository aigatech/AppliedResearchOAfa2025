Old Ye Prompt

This is a website that takes in text prompts and generates an image where the 
prompts are put in context of knighthood. For example, when you put in the prompt "I play videogames at a gaming cafe", an image of a knight playing videogames at a gaming cafe appears. This is the dream website for anyone who has always dreamed of being a knight. The text to image model is the black-forest-labs/FLUX.1-dev model.

In order to run it, you need to download Flask. You will also need to download the transformers library and then the huggingface_hub library. You will also need to download Pillow to process the image data from the AI model. You will also need to install python-dotenv. Once you have the libraries, you just need to call "python submissions/flask_website/app.py" in the terminal and go to the local host address it gives you.

You will also need to create your own .env file where you type HF_TOKEN=YOUR_TOKEN, where YOUR_TOKEN is your own hugging face token.
