from transformers import pipeline
from collections import Counter
from datetime import datetime

class NERAnalyzer:
    def __init__(self, model_name = "dbmdz/bert-large-cased-finetuned-conll03-english", entity_filter=None, batch_size=8, log_file=None):
        """
        Setting up the pipeline for Named Entity Recognition
        """
        self.ner_pipeline = pipeline("ner", model = model_name, aggregation_strategy = "simple")
        self.entity_filter = entity_filter
        self.batch_size = batch_size
        self.log_file = log_file
        if log_file:
            with open(log_file, "w", encoding = "utf-8") as f:
                f.write(f"NER Log at {datetime.now()}\n\n")

    def analyze(self, texts):
        """
        Checks and filters entries in batches
        """
        all_entities = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_entities = self.ner_pipeline(batch)
            filtered_batch = []
            for ents in batch_entities:
                if self.entity_filter:
                    ents = [e for e in ents if e['entity_group'] in self.entity_filter]
                filtered_batch.append(ents)
                if self.log_file:
                    self._log(batch[0], ents)
            all_entities.extend(filtered_batch)
        return all_entities
    
    def _log(self, text, entities):
        with open(self.log_file, "a", encoding = "utf-8") as f:
            f.write(f"[{datetime.now()}] Text: {text[:80]}...\n")
            for e in entities:
                f.write(f" {e['entity_group']}: {e['word']}\n")
            f.write("\n")

    def get_entity_counts(self, entities_batch):
        """
        Returns the amount of word matches in a batch
        """
        counter = Counter()
        for entities in entities_batch:
            for e in entities:
                counter[e['word']] += 1
        return counter