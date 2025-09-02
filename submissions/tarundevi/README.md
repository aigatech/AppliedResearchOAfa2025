# Financial News Market Sentiment Predictor

## Description

This project is a **Gradio-based web application** that analyzes finance-related news articles and predicts the **market sentiment** based on the content. Users can either provide:

- A **URL** of a finance news article  
- Or paste the **text** of an article directly  

The application uses **Hugging Face’s zero-shot classification** (`facebook/bart-large-mnli`) to determine the **article sentiment** (`positive`, `neutral`, `negative`) and maps it to a **market prediction** (`Likely Up`, `Stable`, `Likely Down`).  

---

## Requirements

- Python 3.9+  
- Packages:

## Installation

1. Clone the repository:
    git clone <your-repo-url>
    cd <your-repo-folder>
    python3 -m venv venv
# macOS/Linux
    source venv/bin/activate

# Windows
    venv\Scripts\activate
    pip3 install -r requirements.txt
