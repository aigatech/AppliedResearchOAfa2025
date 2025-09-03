import matplotlib.pyplot as plt
from collections import Counter

def plot_sentiment_distribution(results, save_path = None):
    """
    Plot the distribution of sentiments (positive vs. negative)
    """
    labels = [r["sentiment_label"] for r in results]
    counts = {label: labels.count(label) for label in set(labels)}

    plt.figure(figsize = (6,4))
    plt.bar(counts.keys(), counts.values())
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_entity_frequencies(entity_counts, save_path = None, top_n: int = 20):
    """
    Plot the frequency of the top entities (specified by top_n)
    """
    if not entity_counts:
        print("No entities to plot")
        return
    
    sorted_entities = sorted(entity_counts.items(), key = lambda x: x[1], reverse = True)[:top_n]
    labels, values = zip(*sorted_entities)

    plt.figure(figsize = (10, 5))
    plt.bar(labels, values)
    plt.title(f"Top {top_n} Named Entities")
    plt.xlabel("Entity")
    plt.ylabel("Frequency")
    plt.xticks(rotation = 45, ha = "right")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.show()