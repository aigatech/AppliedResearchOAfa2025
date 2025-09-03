from transformers import pipeline
from datetime import datetime

class SummaryAnalyzer:
    def __init__(self, model_name="facebook/bart-large-cnn", batch_size=4, log_file=None):
        """
        Setting up the pipeline for summarization
        """
        self.summarizer = pipeline("summarization", model=model_name)
        self.batch_size = batch_size
        self.log_file = log_file
        if log_file:
            with open(log_file, "w", encoding = "utf-8") as f:
                f.write(f"Summarization Log at {datetime.now()}\n\n")

    def summarize(self, texts, min_length = 20, max_length = 60):
        """
        Summarize given texts
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            for text in batch:
                if len(text.split()) < 15:
                    summary = text
                    results.append(summary)
                    if self.log_file:
                        self._log(text, summary, skipped = True)
                else:
                    summ = self.summarizer(text, min_length = min_length, max_length = max_length)
                    summary = summ[0]['summary_text']
                    results.append(summary)
                    if self.log_file:
                        self._log(text, summary)
        return results
    
    def _log(self, text, summary, skipped = False):
        with open(self.log_file, "a", encoding = "utf-8") as f:
            status = "SKIPPED" if skipped else "OK"
            compression = len(summary.split()) / len(text.split()) if not skipped else 1.0
            f.write(f"[{datetime.now()}] {status} - compression: {compression:.2f}\n")
            f.write(f"Summary: {summary[:80]}...\n\n")