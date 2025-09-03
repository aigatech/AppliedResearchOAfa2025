# TCGA Tumor Classifier with Explainable AI

## Project Title

**TCGA Tumor Classifier with SHAP & NLP Explanations**

## Overview

This project predicts cancer types based on human gene expression data using an XGBoost classifier. It combines machine learning interpretability with natural language explanations to provide a transparent and educational interface for exploring predictions while showing the user how the machine learning model came to its conclusions.

Key features:

* **Global Feature Importance**: Visualize the most influential human genes across the dataset using XGBoost’s feature\_importances\_.
* **Local Explanations**: Use SHAP values to understand which genes contributed most to a specific prediction.
* **NLP Explanations**: Receive a natural language summary of the predicted cancer type, including risk factors and common treatments, via a HuggingFace GPT model.

Users can either **paste a row of gene expression values** or **select sample data** to explore predictions and explanations interactively.

---

## Before Running

1. **Download mini_tcga.csv at this location: https://drive.google.com/file/d/1eTLBBN1EnjgCnnZRA4VI50GXKsQC9SeA/view?usp=sharing**

2. **Make a folder called data**

3. **Put mini_tcga.csv in this folder called data**

4. **Folder Structure:**

```bash
submission/jayan
│
├─ app.py                 
├─ main.py                
├─ agent.py               
├─ nlp_agent.py           
├─ data/
│   └─ mini_tcga.csv      # small sample dataset (keep it small!)
├─ models/                # saved model artifacts (ignored in submission)
├─ requirements.txt       # all Python dependencies
└─ README.md

```

## How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Train the XGBoost model and save artifacts**:

```bash
python main.py
```

3. **Launch the Gradio web app**:

```bash
python app.py
```

4. **Open the Gradio link in your browser**, input gene expression values or choose an example sample, and explore:

   * Predicted cancer type
   * Class probabilities
   * Global feature importance
   * SHAP-based local explanations
   * Natural language explanation

---

## Notes

* No GPU is required; all components run on CPU.
* HuggingFace models can be swapped for smaller models (DistilBERT, MiniLM, or GPT-OSS) to reduce memory usage.
* The app is for **educational purposes only**. Not intended as medical advice.


---

