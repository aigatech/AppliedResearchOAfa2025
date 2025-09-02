import aiohttp
import os
from dotenv import load_dotenv
from datetime import datetime
from transformers import pipeline

load_dotenv()

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


VALID_CATEGORIES = [
    "business", "entertainment", "general",
    "health", "science", "sports", "technology"
]

def parse_article_date(date_str):
    """Parse date string from NewsAPI into datetime object"""
    if not date_str:
        return datetime.min
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return datetime.min

async def fetch_news(topics=["technology"], max_per_category=5):  
    """Fetch news with AI summaries"""
    API_KEY = os.getenv("NEWSAPI_KEY")
    if not API_KEY:
        raise ValueError("❌ No NewsAPI key found in .env file")
    

    all_articles = []
    
    async with aiohttp.ClientSession() as session:
        for topic in topics:
            url = f"https://newsapi.org/v2/top-headlines?category={topic}&pageSize={max_per_category}&apiKey={API_KEY}"
            
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        error = await response.json()
                        print(f"⚠️ {topic} failed: {error.get('message')}")
                        continue
                        
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    processed_articles = []
                    for article in articles:
                        try:
                            # Generate 2-3 line summary
                            text = f"{article['title']}. {article.get('description', '')}"
                            input_length = len(text.split())
                            max_len = min(80, input_length // 2 + 20)
                            summary = summarizer(text, max_length=max_len, min_length=15, do_sample=False)[0]["summary_text"]

                            processed_articles.append({
                                "title": article["title"],
                                "url": article["url"],
                                "source": article["source"]["name"],
                                "category": topic.upper(),
                                "date": parse_article_date(article.get("publishedAt")),
                                "raw_date": article.get("publishedAt", ""),
                                "summary": summary  # New field
                            })
                        except Exception as e:
                            print(f"⚠️ Failed to process article: {e}")
                            continue
                    
                    processed_articles.sort(key=lambda x: x["date"], reverse=True)
                    all_articles.extend(processed_articles[:max_per_category])
                    
            except Exception as e:
                print(f"⚠️ {topic} error: {str(e)}")
    
    return all_articles
