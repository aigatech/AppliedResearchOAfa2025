# Semantic Search System

Welcome to the **Semantic Search System**!  
This project allows you to store and then search sentences in a database using semantic embeddings. This project implements a semantic search engine using Hugging Face’s **Transformers** and **PyTorch**. It allows users to add, remove, and search sentences based on **semantic similarity** using the `all-MiniLM-L6-v2` model. The program is able to search for semantically similar sentences with **cosine similarity**.

## Features
- **Add**: Store sentences along with their semantic embeddings in the database.  
- **Remove**: Delete sentences and their semantic embeddings from the database.  
- **Search**: Find semantically similar sentences in the database based on a query with adjustable similarity thresholds.  
- **View**: Display all sentences in the database.  
- **Clear**: Remove all sentences from the database.
- **Exit**: Exit the program.

## How to Run

1. Make sure you have **Python 3.8+** installed.  
2. Make sure the following libraries are installed before running the project:
  - `torch`  
  - `transformers`
