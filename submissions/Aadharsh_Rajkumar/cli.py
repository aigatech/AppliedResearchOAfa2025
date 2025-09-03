import argparse
from .analysis.sentiment import SentimentAnalyzer
from .analysis.named_entity import NERAnalyzer
from .analysis.summarizer import SummaryAnalyzer
from .utils.data_loader import load_imdb_dataset
from .utils.save_results import save_to_csv
from .utils.plot_results import plot_sentiment_distribution, plot_entity_frequencies

def run_cli():
    parser = argparse.ArgumentParser(description = "Movie Review Analyzer CLI")
    parser.add_argument(
        "--review",
        type = str,
        required = True,
        help = "Movie review text to analyze (put in quotes)"
    )
    args = parser.parse_args()

    text = args.review

    sentiment_analyzer = SentimentAnalyzer()
    ner_analyzer = NERAnalyzer()
    summarizer = SummaryAnalyzer()

    # for sentiment pipeline
    sentiment = sentiment_analyzer.analyze([text])[0]
    print(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})")

    # for NER pipeline
    entities_batch, _ = ner_analyzer.analyze([text])
    entities = [e['word'] for e in entities_batch[0]] if entities_batch else []
    print(f"Entities found: {', '.join(entities) if entities else 'None'}")

    # for summary pipeline
    summary, ratio = summarizer.summarize([text])[0]
    print(f"Summary: {summary}")
    print(f"Compression ratio: {ratio:.2f}")

if __name__ == "__main__":
    run_cli()