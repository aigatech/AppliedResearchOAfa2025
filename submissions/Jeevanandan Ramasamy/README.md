# 🌐 Lost in Translation Game
By Jeevanandan Ramasamy

## What it does
This project is a **fun AI translation game** built with Hugging Face + Gradio.  
You type a sentence, and the model translates it through multiple languages, showing each step, then back to English.  
The result is a hilarious, sometimes nonsensical, “lost in translation” version of your original text.

Example Input: Hello World!
en → ne: हेलो विश्व !
ne → tr: Hoşgeldin dünya!
tr → zu: Sishayele emhlabeni!
zu → mg: Miaraka amin'izao tontolo izao!
mg → gl: Xunto ao mundo!
gl → en: Together with the world!  

## How to run
1. Clone this repo and `cd` into the folder.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open the local Gradio link in your browser and watch your sentence get hilariously transformed.

## Features
- Translate your sentence across multiple hops in popular languages or all supported languages.
- Step-by-step display of each translation.
- Randomized translation paths for a fresh experience every time.
- Works on CPU or Apple M1 (no GPU required).

## Notes
- Uses facebook/m2m100_418M model.
- English is always the start and end language.
- You can adjust the number of translation hops with the slider.
- ⚠️ Using “all supported languages” may produce weird or less accurate translations.
