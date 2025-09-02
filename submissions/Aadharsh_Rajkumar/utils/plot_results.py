import matplotlib.pyplot as plt
from collections import Counter

def plot_sentiment_distribution(sentiments):
    """
    Plot the distribution of sentiments (positive vs. negative)
    """
    labels = [res['label'] for res in sentiments]
    counter = Counter(labels)

    plt.figure(figsize=(6,4))
    plt.bar(counter.keys(), counter.values(), color = ['green', 'red'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()