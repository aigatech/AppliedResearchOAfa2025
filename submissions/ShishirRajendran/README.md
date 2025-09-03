# GT Sentiment Detoxifier

## What it does
This project connects to Reddit’s **r/gatech** subreddit, analyzes the sentiment of top posts, and replies to posts with highly negative sentiment scores (≥ 0.95).  
When a post is flagged, it uses Hugging Face’s `facebook/bart-large-cnn` model to generate a softer, non-toxic alternative title.  
It also visualizes the distribution of positive vs. negative posts.  

Note: Due to model inaccuracies, the rewritten titles are not always fully “detoxified” and may produce undesirable or awkward phrasing. With the limited time available, improving detoxification quality was not explored deeply.

## How to run it
1. Clone this repository and navigate into the folder.  
2. Install dependencies:  
   ```bash
   pip install praw transformers matplotlib
3. Inside the project make sure to replace parameters, client_id and client_secret to have this run off of your reddit account
4. The num_posts parameter can be used to scrape a different number of posts off of the subreddit for different results.