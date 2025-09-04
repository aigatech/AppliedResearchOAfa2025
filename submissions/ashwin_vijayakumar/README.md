# Semantic FAQ Matcher

The Semantic FAQ Matcher is a Streamlit web app that helps users find the most relevant answer from a list of frequently asked questions (FAQs) using semantic search. Under the hood, the app uses a Hugging Face model to convert text into embeddings. These embeddings capture the semantics of a sentence. Then, the app compares your query vector with precomputed FAQ vectors using **cosine similarity** to find the closest match.

## Dependencies

This project requires the following Python packages:

- **streamlit**: For creating the web interface and handling caching
- **sentence-transformers**: Provides the `all-MiniLM-L6-v2` model for generating sentence embeddings
- **torch (PyTorch) **: Required for tensor operations and cosine similarity calculations
- **transformers**: Hugging Face's transformers library (dependency of sentence-transformers)

You can install all required packages using pip:
```bash
pip install streamlit transformers sentence-transformers torch
```

## Features

- **Semantic Search:** Uses advanced sentence embeddings to match user queries with the most relevant FAQ, even if the wording is different.
- **Instant Results:** Enter your question and get an immediate answer from the FAQ database.
- **Similarity Score:** Displays a similarity score to indicate how closely your query matches the FAQ.
- **Fallback Handling:** If no FAQ matches well enough, the app suggests rephrasing your question.
- **Efficient Performance:** Embeddings are cached for fast and efficient matching.



## How to Use

1. Launch the app using Streamlit:
   ```bash
   streamlit run semantic_faq_matcher.py
   ```
2. Enter your question in the input box.
3. View the matched FAQ, its answer, and the similarity score.
4. If no match is found, try rephrasing your question.
