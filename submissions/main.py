# Load model directly
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

url = input("Enter a financial article url:")
#url = "https://www.investors.com/news/technology/google-stock-google-earnings-google-stock-q22025/"

response =requests.get(url)
soup=BeautifulSoup(response.text,"html.parser")

paragraphs = [p.get_text() for p in soup.find_all("p")]
article_text = " ".join(paragraphs)

chunks = [article_text[i:i+500] for i in range(0, len(article_text), 500)]


results= nlp(chunks)
# Example
#text = "The company reported strong earnings growth despite market volatility."
for i, res in enumerate(results):
    print(f"Chunk {i+1}: {res}")
labels = [r['label'] for r in results]
print("Overall sentiment:", max(set(labels), key=labels.count))