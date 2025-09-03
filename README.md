Hugging Face Project

Author: Shaunak Kasa

This project is a simple command-line demo that showcases several Natural Language Processing (NLP) tasks using the Hugging Face Transformers
library.

Users can interactively choose between:

Sentiment Analysis

Summarization

Translation (English → French / Spanish)


Dependencies

PyTorch – deep learning backend

Transformers – Hugging Face models and pipelines

SentencePiece – required for MarianMT translation models

Protobuf – required by Transformers


Models Used

Sentiment Analysis: distilbert-base-uncased-finetuned-sst-2-english

Summarization: facebook/bart-large-cnn

Translation (EN → FR): Helsinki-NLP/opus-mt-en-fr

Translation (EN → ES): Helsinki-NLP/opus-mt-en-es


How to Run

Make sure you have Python 3.10+ installed.

Install the required libraries:

pip install transformers==4.44.2
pip install torch==2.4.1
pip install sentencepiece==0.2.0
pip install protobuf==5.28.2


Run the program:

python main.py


Choose a task from the menu:

1 → Sentiment Analysis

2 → Summarization

3 → Translation (English → French / Spanish)

q → Quit
