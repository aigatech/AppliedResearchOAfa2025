# Reddy Bot

**Reddy Bot** (created by *Aditya Jha*) is a command-line tool that analyzes the sentiment of Reddit threads using a pre-trained language model.

---

## Features

- Input a Reddit thread URL via the command line
- Scrapes the main post and top-level comments
- Splits each comment into individual sentences
- Uses Hugging Face's **DistilBERT** for sentiment analysis
- Classifies each sentence as **positive** or **negative**
- Outputs summary statistics and an overall sentiment classification

---

## How It Works

1. You provide a Reddit thread URL.
2. Reddy Bot fetches the main post and top-level comments.
3. It splits the text into individual sentences.
4. Each sentence is analyzed for sentiment using Hugging Face’s **DistilBERT** model.
5. Sentiment results are aggregated to calculate positive vs. negative percentages.
6. Reddy Bot determines whether the overall tone is positive, negative, or neutral/mixed.

---

## Design Considerations

- **Efficiency**: Reddy Bot only analyzes the main post and top-level comments. Deep threads can have tens of thousands of comments, which would be impractical to analyze in full.

- **Sentence-Level Analysis**: Each comment is broken into sentences and analyzed independently. This allows the tool to accurately capture mixed opinions and subtle shifts in tone within a single comment.

- **URL Validation**: Basic validation is included to ensure the input is a Reddit thread URL. Malformed or unrelated URLs may still cause errors, so paste the URL directly from Reddit.

---

## How to Run

> Requires **Python 3.10 or higher**  
> The model will be downloaded on first use (approx. 60MB), which may take a few seconds.

### Steps:

1. Open a terminal (Mac/Linux) or Command Prompt (Windows)
2. Navigate to the folder containing `main.py` (this folder also contains this `README.md`)
3. Run the script using:

   ```bash
   python main.py
