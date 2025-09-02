# Financial News Market Sentiment Predictor
Description

This project is a Gradio-based web application that analyzes finance-related news articles and predicts the market sentiment based on the content. Users can either provide:

A URL of a finance news article

Or paste the text of an article directly

The application uses Hugging Face’s zero-shot classification (facebook/bart-large-mnli) to determine the article sentiment (positive, neutral, negative) and maps it to a market prediction (Likely Up, Stable, Likely Down).

# # Instructions
1. python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
2. pip install -r requirements.txt
3. python main.py
4. Open the Gradio UI in your browser.
5. Select the input type:
URL – paste a finance news article link
Text – paste the article text directly
6. Click Predict to see:
    Article Sentiment (positive, neutral, negative)
    Market Prediction (Likely Up, Stable, Likely Down)

