# ReadSlice

## What it does
ReadSlice takes any research paper or technical blog from the internet and helps you remember what you learned by generating:
- Flashcard-style questions from each section
- Concise summaries you can revisit later

It uses BeautifulSoup for scraping and HuggingFace's FLAN-T5 Small** model for summarization.

---

## Why this project?
As an ML research assistant at Georgia Tech, I spend a lot of time reading papers for literature reviews—whether to identify gaps, analyze limitations, or learn about new techniques.

The problem:
- It's easy to forget what you've read, leading to wasted time revisiting papers.
- You can fall into a "reading coma"—absorbing a lot of text but retaining very little.

**ReadSlice** helps by:
- Forcing active recall with flashcard/quizlet-style questions
- Providing short summaries of each section for quick review

---

## How to run

git clone https://github.com/yourusername/ReadSlice.git
cd ReadSlice

python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

python main.py

In main.py you can change the link on line 6 to any paper or technical blog! 