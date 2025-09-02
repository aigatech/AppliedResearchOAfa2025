# Financial News Market Sentiment Predictor

## Description

This project is a **Gradio-based web application** that analyzes finance-related news articles and predicts the **market sentiment** based on the content. Users can either provide:

- A **URL** of a finance news article  
- Or paste the **text** of an article directly or type a few sentences manually  

The application uses **Hugging Face’s zero-shot classification** (`facebook/bart-large-mnli`) to determine the **article sentiment** (`positive`, `neutral`, `negative`) and maps it to a **market prediction** (`Likely Up`, `Stable`, `Likely Down`).  

# 📈 Financial News Market Sentiment Predictor

This project uses [Hugging Face Transformers](https://huggingface.co/transformers/) and [Gradio](https://gradio.app/) to predict **market sentiment** from financial news articles.  
You can provide either a **URL** (the script will scrape and analyze the article text) or paste the **article text** directly.  

The model used is: [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli).  

---

## 🚀 Features
- Extracts article text automatically from a **URL**.
- Classifies sentiment as **positive, neutral, or negative**.
- Maps sentiment to market prediction:  
  - ✅ Positive → *Likely Up*  
  - ⚠️ Neutral → *Stable*  
  - ❌ Negative → *Likely Down*  
- Interactive **Gradio web app**.

---

## 🔧 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/financial-sentiment.git
```

Set up an environment

```
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

Install the required libraries

```
pip install -r requirements.txt
```
Then, run 
```
python main.py
```
You should see output like 
```
Running on local URL:  http://127.0.0.1:7860
```

