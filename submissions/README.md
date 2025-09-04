# Research Paper Explainer

This repository contains a simple Gradio app that summarizes research text using an instruction-following FLAN-T5-small model from Hugging Face. Paste text, adjust the Sophistication slider, and get a summary whose tone and length adapt—from ELI5 to expert (longer at higher sophistication). While I had orignally intended to parse full research papers, python libraries for this task seem to be too slow to use for a quick demo!

## Environment Setup

1. **Create and activate a virtual environment** - I use python venv's on Windows, which can be setup as follows in the terminal:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate     #On mac, I believe it is: source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Launch the Gradio interface with:
```bash
python app.py
```
The terminal will show a local URL (e.g. `http://127.0.0.1:7860`). Open it in your browser to use the summarizer.

## Notes
- The model weights are downloaded automatically from Hugging Face on first run.
- The app runs on CPU by default but will use a GPU if available.
