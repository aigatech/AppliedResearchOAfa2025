from transformers import pipeline
from datetime import datetime

class SentimentAnalyzer:
    def __init__(self, model_name = "distilbert-base-uncased-finetuned-sst-2-english", batch_size=8, confidence_threshold=0.0, log_file=None):
        """
        Setting up the pipeline for sentiment analysis
        """
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True, 
            max_length=512    
        )
        self.batch_size = batch_size
        self.conf_threshold = confidence_threshold
        self.log_file = log_file
        if log_file:
            with open(self.log_file, "w", encoding = "utf-8") as f:
                f.write(f"Sentiment analysis at {datetime.now()}\n\n")
    
    def analyze(self, texts):
        """
        Checks for entries in batches and returns predictions above the confidence threshold
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_results = self.sentiment_pipeline(batch)
            for text, res in zip(batch, batch_results):
                if res['score'] >= self.conf_threshold:
                    results.append(res)
                    if self.log_file:
                        self._log(text, res)
                else:
                    low_conf_res = res.copy()
                    low_conf_res['label'] = f"{res['label']} (low confidence)"
                    results.append(low_conf_res)
                    if self.log_file:
                        self._log(text, low_conf_res, low_conf=True)
        return results
        
    def _log(self, text, result, low_conf=False):
        with open(self.log_file, "a", encoding="utf-8") as f:
            status = "LOW_CONFIDENCE" if low_conf else "OK"
            f.write(f"[{datetime.now()}] {status} - {result['label']} ({result['score']:.2f}): {text[:80]}...\n")