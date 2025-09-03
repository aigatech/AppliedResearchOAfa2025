#Explainable Person Detection for Safety in Warehouses

## What it does
This project uses HuggingFace’s **YOLOS** (You Only Look One-level Series) model to detect people in warehouse settings.
It then applies a simple safety rule:
- **Unsafe** if a **person** is detected in the image 
- **Safe** if no person is detected  

To improve interpretability, the project also uses **Flan-T5** to generate a natural-language explanation of the classification.  

This mimics a **two-system cognitive framework**:
- **System 1 (Fast):** YOLOS for object detection  
- **System 2 (Slow):** Flan-T5 for reasoning and explanation 

I tried to tie in the work I did on the two projects at my internship: Computer Vision to detect individuals near warehouse machinary, and RAG based LLMs.

## How to run
1. Install dependencies:
   pip install transformers pillow torch
2. Run the script with either your own image or the images attached in this folder with: 
    python explainable_yolo.py --image path/to/image.jpg

Example output:
For sample.jpg the output was:
Prediction: SAFE
Explanation: No person was detected.
