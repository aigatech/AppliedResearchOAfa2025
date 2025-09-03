#Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import requests
from bs4 import BeautifulSoup
import json

#Load pre-trained model and tokenizer - DistilBERT
print("⏳ Loading model... Please wait 🤖💭 \n")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

#Function to get sentences from a paragraph and return them as a list
def get_sentences(paragraph):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(paragraph)
    return [s for s in sentences if s != '']

#Function to ensure a URL exists and is valid *not working right now)
def validate_reddit_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; url-validator/1.0)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"Failed to reach the URL: {e}")
        return False

    if response.status_code == 200:
        # Optionally, check if it's actually a Reddit thread by looking for expected content
        if "reddit.com" not in response.url:
            print("URL does not appear to be a Reddit URL.")
            return False
        return True
    elif response.status_code == 404:
        print("The Reddit thread does not exist (404).")
        return False
    else:
        print(f"Failed to access the Reddit thread. HTTP status code: {response.status_code}")
        return False

#Function to get main and top comments from a Reddit thread given its URL, returned in a list
def get_reddit_thread_paragraphs(thread_url):
    if not thread_url.endswith(".json"):
        if thread_url.endswith("/"):
            thread_url += ".json"
        else:
            thread_url += "/.json"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; sentiment-analyzer/1.0)"
    }

    response = requests.get(thread_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch thread: Status code {response.status_code}")

    data = response.json()

    try:
        post_body = data[0]["data"]["children"][0]["data"]["selftext"]
    except Exception:
        post_body = ""

    comments = []
    try:
        comment_items = data[1]["data"]["children"]
        for comment in comment_items:
            if comment["kind"] != "t1":
                continue 
            body = comment["data"].get("body")
            if body and body != "[deleted]" and body != "[removed]":
                comments.append(body)
    except Exception:
        pass

    paragraphs = [post_body] + comments if post_body else comments

    return paragraphs

#Function to analyze the sentiment of given text using Hugging Face's DistilBERT model
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad(): 
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "positive" if predicted_class == 1 else "negative"
    return sentiment

#Function to calculate the percentage of positive/negative sentiments
def calculate_percentage(part, whole):
    if whole == 0:
        return 0
    return round((part / whole) * 100, 2)

#main function to run the command line input and provide output
def main():
    print("👋 Welcome! Reddy Bot (created by Aditya Jha) takes the URL of a Reddit thread, analyzes the sentiment of its main and top comments, and tells you how positive or negative the thread is. 💬📈 \n")
    print("🔍 You can use Reddy Bot to see how positive/negative a Reddit thread is, and do research on the sentiment of communities on Reddit. 🧠📊 \n")
    thread_url = input("Copy and paste the URL of Reddit thread to analyze (Please include https://): ")
    validate_reddit_url(thread_url)
    comments = get_reddit_thread_paragraphs(thread_url)

    positives = 0
    negatives = 0

    number_of_sentences = 0

    #analyze by sentence
    for comment in comments:
        for sentence in get_sentences(comment):
            sentiment = analyze_sentiment(sentence)
            number_of_sentences+=1
            if sentiment == "positive":
                positives += 1
            else:
                negatives += 1

    #printing results
    print(f"📊 Reddy Bot thinks the given Reddit thread was: {calculate_percentage(positives, number_of_sentences)}% positive and {calculate_percentage(negatives, number_of_sentences)}% negative. \n")

    if calculate_percentage(positives, number_of_sentences) > 0.60:
        print("Overall sentiment: Positive. The positive views outweigh the negative views. ✅")
    elif calculate_percentage(negatives, number_of_sentences) > 0.60:
        print("Overall sentiment: Negative. The negative views outweigh the positive views. ❌")
    else:
        print("Overall sentiment: Neutral / Mixed. Your thread has a healthy mix of positive and negative viewpoints. ⚖️")

#Run main
if __name__ == "__main__":
    while True:
        main()
        again = input("🔁 Would you like Reddy Bot to analyze another Reddit thread? (y/n): ").strip().lower()
        if again != "y":
            print("👋 Thank you for using Reddy Bot! Goodbye. 👋 \n")
            break
