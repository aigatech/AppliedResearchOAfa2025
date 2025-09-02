# 🌱 Plant Disease Detector

## What it does
This app allows users to upload one or more plant images and predicts potential diseases using a Vision Transformer (ViT) model from Hugging Face.  
It highlights disease likelihood with colored bars, provides a health status (healthy, diseased, or uncertain), and gives some recommendations for plant care.

## How to run it
1. Clone the repository:

```bash
git clone <https://github.com/samik-py/AppliedResearchOAfa2025>
cd submissions
```

2. Create virtual environment (.venv), preferably through cmd + shift + p => "Python: create environment" in VsCode or through the terminal. Make sure to activate the environment by running

```bash

#activate
source .venv/bin/activate

```
(on mac, windows may be different)

3. Install dependencies. Run these lines in the terminal (assuming pip is already installed)

```bash

pip install pillow torch torchvision torchaudio transformers numpy opencv-python matplotlib
pip install gradio==3.39.0

```

4. Run the program. In the terminal, there should be a gradio link to a temporary hosting of the web app. Upload any images of plants to get 
a determination of whether or not the plant is sick, steps to improve it's health, and a heatmap of the affected areas.


