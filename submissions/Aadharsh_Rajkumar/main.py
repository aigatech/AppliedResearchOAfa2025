from submissions.Aadharsh_Rajkumar.utils.data_loader import load_imdb_dataset
from submissions.Aadharsh_Rajkumar.utils.save_results import save_to_csv
from submissions.Aadharsh_Rajkumar.utils.plot_results import plot_sentiment_distribution, plot_entity_frequencies

from submissions.Aadharsh_Rajkumar.analysis.sentiment import SentimentAnalyzer
from submissions.Aadharsh_Rajkumar.analysis.named_entity import NERAnalyzer
from submissions.Aadharsh_Rajkumar.analysis.summarizer import SummaryAnalyzer

import os
from datetime import datetime

def run_analysis(num_samples: int = 200, output_dir: str = "outputs", plot: bool = True):
    """
    Do sentiment analysis, NER, and summarization on the IMDB reviews.
    Save results to a CSV and generate the plots.
    """
    dataset = load_imdb_dataset(split = "test")
    texts = dataset["text"][:num_samples]

    sentiment_analyzer = SentimentAnalyzer()
    ner_analyzer = NERAnalyzer()
    summarizer = SummaryAnalyzer()

    results = []
    all_entities = {}

    for i, text in enumerate(texts):
        print(f"\n--- Review {i+1}/{num_samples} ---")
        sentiment = sentiment_analyzer.analyze([text])[0]
        entities_batch, entity_counts = ner_analyzer.analyze([text])
        entities = [e['word'] for e in entities_batch[0]] if entities_batch else []
        summary, ratio = summarizer.summarize([text])[0]

        for ent in entities:
            all_entities[ent] = all_entities.get(ent, 0) + 1

        result = {
            "review": text,
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "entities": ", ".join(entities) if entities else "None",  # ✅ now entities is a list of strings
            "summary": summary,
            "summary_compression_ratio": ratio
        }
        results.append(result)

    os.makedirs(output_dir, exist_ok = True)
    
    # saving to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"analysis_results_{timestamp}.csv")
    save_to_csv(results, csv_path)

    # plotting
    if plot:
        plot_sentiment_distribution(results, os.path.join(output_dir, f"sentiment_{timestamp}.png"))
        plot_entity_frequencies(all_entities, os.path.join(output_dir, f"entities_{timestamp}.png"))

if __name__ == "__main__":
    run_analysis(num_samples = 50, plot = True)