import ddgs
import requests
from transformers import pipeline
from newspaper import Article
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

summarizer = pipeline("summarization", model="google/pegasus-xsum")

FOOTBALL_SITES = ["cbssports.com", "foxsports.com", "si.com", "sports.yahoo.com"]

def search_article(query):
    """Search for query and return article url"""
    with ddgs.DDGS() as ddgs_client:
        results = ddgs_client.text(query, region='wt-wt', safesearch='Off', max_results=10)
        for r in results:
            url = r["href"]
            if any(site in url for site in FOOTBALL_SITES):
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    if article.text.strip():
                        return url
                except:
                    continue
        return None

def fetch_article(url):
    """Download text and image from article url"""
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    main_image = article.top_image
    return text, main_image

def summarize_text(text):
    """Generate caption"""
    MAX_CHARS = 3000
    text = text[:MAX_CHARS]
    text = "Write a short Instagram caption about the following football game based on the article:\n" + text
    summary = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=False
    )
    sentence_list = summary[0]["summary_text"].strip().split(".")
    summary_text = ""
    for sentence in sentence_list:
        sentence = sentence.strip()
        if sentence:
            summary_text += sentence + ". "
    return summary_text.strip()

def main():
    topic = input("Enter football topic (e.g. 'Georgia Tech wins against Colorado'): ")
    
    url = search_article(topic + " football")
    if not url:
        print("No football news article found, try another topic.")
        return

    print("Found article:", url)

    text, main_image = fetch_article(url)
    print("\nOriginal length:", len(text), "characters")

    caption = summarize_text(text)
    print("\nCaption:\n", caption)
    print("\nImage URL:\n", main_image)

    img_data = requests.get(main_image).content
    with open('post_image.jpg', 'wb') as handler:
        handler.write(img_data)

if __name__ == "__main__":
    main()
