# Wikipedia Path Finder

Wikipedia races are a fun game to play with friends. Starting at any Wikipedia page, you must navigate to another specific page by only clicking Wikipedia links.

This seems daunting, especially with end pages that could not be more different (Magneto to Chandragupta Maurya???)

With the Wikipedia Path Finder, simply plug in your start and end pages and find the best path between them in seconds! Easily beat your friends without thinking too hard or cheating.

## How It Works
The tool grabs all the links on the start page and uses a MiniLM model to calculate the embeddings of each link. It then ranks these based on similarity to the end page and performs a Breadth First Search to find the optimal path.

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
python -m streamlit run .\wiki_path_finder.py
```
