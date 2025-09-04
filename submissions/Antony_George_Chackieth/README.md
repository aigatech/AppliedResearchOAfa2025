# Sparky — Roast, Coach, & Speak (HuggingFace)

What it is
- A small app that roasts playfully with an original robot persona and speaks via TTS.

How it works
- Text model (preferred): Qwen/Qwen2.5-1.5B-Instruct (falls back to distilgpt2)
- TTS: microsoft/speecht5_tts + microsoft/speecht5_hifigan (random speaker embedding)

Run
python -m venv venv
source venv/bin/activate 
pip install --upgrade pip
pip install -r requirements.txt
python app.py

Notes
- First run downloads models to HF cache.
- No model weights or venv committed.