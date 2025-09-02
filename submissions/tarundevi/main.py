# main.py
from newspaper import Article
from transformers import pipeline
import gradio as gr


# Initialize Classifier

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Functions
def extract_article(url):
    """
    Extracts the main content of a news article from a given URL.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text and predicts market sentiment.
    """
    labels = ["positive", "neutral", "negative"]
    result = classifier(text, candidate_labels=labels)
    sentiment = result['labels'][0]
    
    if sentiment == "positive":
        market_sentiment = "Likely Up"
    elif sentiment == "negative":
        market_sentiment = "Likely Down"
    else:
        market_sentiment = "Stable"

    return sentiment, market_sentiment

def predict_market(input_type, input_content):
    """
    Gradio function: input_type: 'URL' or 'Text', input_content: URL or text
    """
    if input_type == "URL":
        try:
            article_text = extract_article(input_content)
        except Exception as e:
            return f"Error extracting article: {e}", ""
    else:
        article_text = input_content

    sentiment, market_sentiment = analyze_sentiment(article_text)
    return sentiment, market_sentiment

# Gradio Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Financial News Market Sentiment Predictor")
        gr.Markdown("Paste a finance news **URL** or the **text** of an article and get a market sentiment prediction.")

        input_type = gr.Radio(["URL", "Text"], label="Select input type")
        input_content = gr.Textbox(label="Enter URL or paste text here", lines=8, placeholder="Enter article URL or paste article text...")
        sentiment_output = gr.Textbox(label="Article Sentiment")
        market_output = gr.Textbox(label="Market Prediction")

        submit_btn = gr.Button("Predict")
        submit_btn.click(predict_market, inputs=[input_type, input_content], outputs=[sentiment_output, market_output])

    demo.launch()

if __name__ == "__main__":
    main()
