# Hybrid Adversarial Evolver

## Overview
This project evaluates a language model on a Q&A dataset for the AI club’s 1.5-hour interview demo. It uses `microsoft/phi-2` (~2.7B parameters) with Hugging Face’s `transformers` to process `data/seed.jsonl`, running on an ASUS ROG Zephyrus G14 (RTX 4070, 8GB VRAM, 32GB RAM, Windows).

## Requirements
- Python 3.8–3.11
- Libraries: `transformers`, `accelerate`, `torch`, `datasets`
- Install:
pip install transformers accelerate torch datasets
text- Dataset: `data/seed.jsonl` (Q&A pairs in JSONL format)

## Setup
1. Create a virtual environment:
python -m venv venv
.\venv\Scripts\activate
text2. Install dependencies:
pip install transformers accelerate torch datasets
text3. Ensure `data/seed.jsonl` exists, e.g.:
```json
{"question": "What is 2+2?", "answer": "4"}
{"question": "What is the capital of France?", "answer": "Paris"}