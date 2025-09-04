# Semantic FAQ Search Engine

A FAQ search engine that uses semantic similarity to find relevant answers, even when questions are worded differently.

## How It Works

Uses HuggingFace sentence transformers to convert text into embeddings and cosine similarity to find the best matching FAQ answer. Understands meaning, not just exact word matches.

Example:
- FAQ: "What is your return policy?"
- Query: "How do I get my money back?"
- Result: Correctly identifies these as similar and returns the return policy answer.

## Features

- Simple Streamlit web interface
- Custom FAQ input in Q: and A: format
- Sample FAQ data included
- Semantic understanding of questions
- CPU-efficient model (all-MiniLM-L6-v2)
- Similarity scoring
- Returns top 3 relevant answers

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

Launch the web app and:
- Click "Load Sample FAQ" to get started
- Input your FAQ data in Q: and A: format
- Ask questions in natural language
- Get semantic search results with similarity scores

## Technical Details

- Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
- Similarity: Cosine similarity
- Performance: Fast on CPU, no GPU required
