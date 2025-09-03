# 🌱 Plant Disease Detector

## What it does
This Plant Disease Detector allows users to upload plant images and predict potential diseases using a Vision Transformer (ViT) model from Hugging Face.

It highlights disease likelihood, provides a health status (healthy, diseased, or uncertain), and gives some recommendations for plant care.

Also, the program provides a heat-map overlay of the original image to see the problem areas on the plant!

## How to run it (python 3+)
1. Clone the repository:

```bash
git clone <https://github.com/samik-py/AppliedResearchOAfa2025>
cd submissions
```

2. Initialize and activate the virtual environment. Either through VsCode's cmd + shift + p => "Python create virtual environment" or by running:

```bash

#make environment
python3 -m venv venv
#activate
source .venv/bin/activate

```
(remember to select the correct python interpreter)

(on mac, windows may be different)

3. Install dependencies. Run these lines in the terminal (assuming pip is already installed)

```bash

pip install pillow torch torchvision torchaudio transformers numpy opencv-python matplotlib
pip install gradio==3.39.0

```

4. Run the program. In the terminal, there should be a gradio link to a temporary hosting of the web app. The first time you run the program, it may take up a minute to set up the hosting. When it loads, upload any images of plants to get a determination of whether or not the plant is sick, steps to improve it's health, and a heatmap of the affected areas.


