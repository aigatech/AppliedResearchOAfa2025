# Bias Buster: Detecting Sentiment Bias in Headlines

## What it does
Analyzes news headlines from different sources (CNN, Fox News, BBC) to detect sentiment patterns that might indicate bias. Uses HuggingFace's sentiment analysis model to classify headlines as positive or negative.

## How to run it
1. Install requirements:
pip install transformers
2. Run the script:
python sentiment_detector.py

## What you'll see
The program will show the count of sentiments for each news source in a form of 

CNN: {'Positive': #, 'Negative': #}
Fox News: {'Positive': #, 'Negative': #} 
BBC: {'Positive': #, 'Negative': #}

## Why this matters
The way headlines are made may have different editorial leanings and can inturn have bias. This script detects those patterns. 

## Details
This script uses the model distilbert-base-uncased-finetuned-sst-2 sentiment classifier

The script analyzes 12 recent (9/3/2025) headlines from 3 different new sources