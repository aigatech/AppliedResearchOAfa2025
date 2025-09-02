# PDF Semantic Search

## What it does
A lightweight semantic search engine for PDFs using Hugging Face embeddings (`intfloat/e5-small`).

- Extracts text from PDFs
- Embeds documents with Hugging Face models
- Runs semantic search against queries
- Returns top matches with similarity score and text preview

## How to run
1. Install dependencies:
   ```bash
   pip install torch transformers pymupdf
2. Add the desired PDFs in `pdf_files` in lines 73
3. Run the program and follow the instructions in the console. 
