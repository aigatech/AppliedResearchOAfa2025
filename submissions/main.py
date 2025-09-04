
from transformers import pipeline
import matplotlib.pyplot as plt

def analyze_reviews(reviews):
    #Use distilbert model for sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = sentiment_analyzer(reviews)

    # Print individual results for each review
    print("\nIndividual Sentiment Results:")
    for review, result in zip(reviews, results):
        print(f" - \"{review}\" -> {result['label']} (confidence {result['score']:.2f})")

    # Count distribution
    counts = {"POSITIVE": 0, "NEGATIVE": 0}
    for r in results:
        counts[r["label"]] += 1

    # Plot pie chart
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ["#66bb6a", "#ef5350"]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title("Sentiment Distribution of Course Reviews")
    plt.show()


def main():
    print("GT Course Review Sentiment Analyzer")
    reviews = []

    while True:
        text = input("Enter a course review or type 'done' to finish: ")
        if text.lower() == "done":
            break
        if text.strip():
            reviews.append(text)

    if reviews:
        analyze_reviews(reviews)
    else:
        print("No reviews entered. Exiting.")


if __name__ == "__main__":
    main()
