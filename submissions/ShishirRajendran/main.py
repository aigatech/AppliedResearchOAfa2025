import praw
from transformers import pipeline
import matplotlib.pyplot as plt

reddit = praw.Reddit(
    client_id="pjUXdgDGKYimj9e_7H63zA",
    client_secret="-U2UaxmMGK9StcLGB4QD0HDqjQI8cA",
    user_agent="GT Sentiment Project"
)

sentiment = pipeline("sentiment-analysis", return_all_scores=True)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def fetch_posts(limit=20):
    return [s for s in reddit.subreddit("gatech").top(time_filter="all", limit=limit) if not s.stickied]


def analyze_and_reply(posts):
    reply_to_posts = []
    for submission in posts:
        results = sentiment(submission.title)[0]
        negative_score = next(item['score'] for item in results if item['label'] == "NEGATIVE")
        if negative_score >= 0.95:
            new_title = summarizer(submission.title, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
            reply_text = f"Next time, use a non-toxic title like this instead: '{new_title}'"
            reply_to_posts.append((submission, reply_text))
    return reply_to_posts


def visualize_sentiment(posts):
    all_results = sentiment([s.title for s in posts])
    labels = [max(r, key=lambda x: x['score'])['label'] for r in all_results]
    pos_count = labels.count("POSITIVE")
    neg_count = labels.count("NEGATIVE")

    plt.bar(["Positive", "Negative"], [pos_count, neg_count], color=["green", "red"])
    plt.title("Sentiment of r/gatech Posts")
    plt.ylabel("Count")
    plt.show()


def main(num_posts=20):
    posts = fetch_posts(limit=num_posts)
    replies = analyze_and_reply(posts)

    for submission, reply in replies:
        print(f"Would reply to: {submission.title}")
        print("Reply:", reply)

    visualize_sentiment(posts)


if __name__ == "__main__":
    main(20)
