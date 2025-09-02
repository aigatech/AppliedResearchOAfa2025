## Project Title
How did 8 years of medical school not teach you how to write?

OR

Medical Report Standardizier/Summarizer

## What It Does
During my AI research internship at the Stanford Center for Artificial Intelligence in Medicine and Imaging, I observed that medical professionals, especially radiologists, often struggle with long, messy, and unstructured clinical notes. These notes may include abbreviations, shorthand, inconsistent terminology, and scattered information, making it difficult to share information efficiently.

This project addresses that challenge by providing a Medical Report Standardizer & Summarizer using HuggingFace's biomedical-ner-all model. The system performs several key steps:

**Abbreviation Expansion**: Automatically expands common clinical abbreviations (e.g., "Pt" → "Patient", "c/o" → "complains of") to improve readability and  accuracy.

**Named Entity Recognition (NER)**: Uses a pretrained biomedical NER model to identify and classify entities such as symptoms, diseases, medications, procedures, and body structures.

**Subword Token Merging**: Combines split tokens produced by the model (e.g., "met" + "##oprolol" → "metoprolol") to produce complete terms.

**Standardization**: Maps the model's raw entity labels to a consistent schema, creating structured outputs.

**Summary Generation**: Produces a concise summary sentence to allow clinicians to quickly grasp the most relevant information.

## How to Run It
Install dependencies:

pip install transformers gradio  
python main.py
