# BLIP Image Captioner with Retrieval-Augmented Generation

## Overview

This project is v2 of the submission and my approach to learning how to implement RAG-like behavior at a smaller scale: a web app that lets users upload images to get automatically generated captions enriched with text retrieved from a large custom corpus. The enrichment process uses a persistent FAISS vector store and a T5-small text generator to rewrite enriched captions for a polished user experience. Sentiment analysis provides emotional context for each enriched caption.

## Features

- Image captioning with Salesforce BLIP
- Retrieval-augmented enrichment from a large text corpus via FAISS
- Caption rewriting using T5-small for fluency and detail
- Sentiment analysis on enriched caption
- Lightweight, CPU-friendly Gradio web app interface

## Setup Instructions

1. Clone or fork the repo and navigate into the submission directory:

```

cd submissions/salman_guliwala

```

2. Install dependencies:

```

pip install -r requirements.txt

```

3. Review and/or edit `corpus.txt`. It contains humorous witty phrases used to enrich captions.

4. Build the FAISS index from `corpus.txt` (run once):

```

python faiss_build.py

```

5. Run the application:

```

python image_captioner.py

```

6. Open the local URL displayed in the terminal (usually `http://localhost:7860`) in your browser.

7. Upload any image and generate captions along with a corresponding sentiment analysis.

## File Descriptions

- `corpus.txt`: Text corpus of facts/phrases used for retrieval enrichment.
- `faiss_build.py`: Script to build the FAISS vector search index from `corpus.txt`.
- `image_captioner.py`: Main app script running BLIP captioning, RAG enrichment, rewriting, and sentiment analysis with Gradio interface.
- `requirements.txt`: Required Python packages.
- `.gitignore`: Files excluded from git commits.

## Models Used

- Salesforce BLIP (image captioning)
- Sentence Transformers all-MiniLM-L6-v2 (embedding for retrieval)
- T5-small (caption rewriting)
- Cardiff NLP's RoBERTa (sentiment analysis)

## Notes

This app runs fully on CPU without requiring GPUs. The FAISS vector store optimizes retrieving relevant facts quickly for caption enrichment.

---

Feel free to reach out with any questions or suggestions!

```

```
