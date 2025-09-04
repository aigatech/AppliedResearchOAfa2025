# Simple Robot Vision Captioning (Office Desk Only)

## Introduction
Hi! My name is **Arjun Sood**, and this is a simple AI app that:
1) generates a short caption for an uploaded **office-desk** image using BLIP, and  
2) extracts **grabbable office items** mentioned in that caption using a small built-in lexicon.

> **Notes**
> - **CPU-only**; no GPU or CUDA required.  
> - The app displays the caption and detected office items (there is **no inference-time display**).

---

## How to Run

> Commands shown for Windows PowerShell. On macOS/Linux, use your shell’s venv activation instead.

1. **Create a project folder**
   ```powershell
   mkdir robot-vision-captioning
   cd robot-vision-captioning

2. **Create and activate a virtual environment**
   python -m venv .venv
   .venv\Scripts\Activate.ps1

3. **Upgrade pip**
   python -m pip install --upgrade pip

4. **Install Dependencies**
   pip install "transformers>=4.41" torch pillow "gradio>=4.44"

5. **Save the app**
   Put your Python script in this folder as app.py

6. **Run the app**
   python app.py

7. **Open in your browser**
   After the BLIP model downloads from Hugging Face, open the local URL printed in the terminal. For example:

   Running on local URL: http://127.0.0.7860

8. **Try the app**
   Upload an office-desk photo (there are examples in the images folder) and click Caption Image
   You'll see the generated caption and the grabbable office items detected from the caption

## Future Work (Ideas to improve)

1. Generate several captions and pick the best using a image-text score
2. Draw boxes around detected items on the image using an open-vocabulary detector (so users can see where each object is).
3. Let users add/remove items in the object list from the UI and save their custom list for next time.