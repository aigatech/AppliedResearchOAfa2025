import csv

def save_to_csv(filename, texts, sentiments, summaries):
    """
    Save results of analysis to a CSV file (-save tag) 
    """
    with open(filename, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Sentiment", "Score", "Summary"])
        for text, sentiment, summary in zip(texts, sentiments, summaries):
            writer.writerow([text, sentiment['label'], sentiment['score'], summary])
    
    print(f"Results have been saved to {filename}")
