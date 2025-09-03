# Football Caption + Image Finder

## What it does
This project takes a football games and topics like "Georgia Tech wins against Colorado" and searches for a recent article from trusted football news sources. It then generates:  
- Short Instagram-style caption relevant to the game using a Hugging Face LLM (Pegasus-XSum)  
- The main image URL from the article for social media posts  

Helps users create quick AI-assisted social media post for football games after big wins.

Outputs: Reference article URL, Image URL, caption, and downloaded .jpg of image in current folder

## How to run it
1. Python 3.8+
2. Install dependencies with: 
pip install transformers newspaper3k ddgs requests lxml_html_clean
3. Run program with:
python main.py
