import sqlite3
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SemanticSearch:
    def __init__(self):
        """Initialize the semantic search system."""
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sentences = []
        self.sentence_embeddings = torch.empty((0,))

        self.init_db()  # Initialize the database

    def init_db(self):
        """Initialize the database and create necessary tables."""
        with sqlite3.connect('semantic_search.db') as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS sentences (
                    id INTEGER PRIMARY KEY,
                    text TEXT UNIQUE,
                    embedding BLOB
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY,
                    query TEXT
                )
            ''')
            conn.commit()

    def save_to_db(self, sentence, embedding):
        """Save a sentence and its embedding to the database."""
        with sqlite3.connect('semantic_search.db') as conn:
            c = conn.cursor()
            try:
                c.execute('INSERT INTO sentences (text, embedding) VALUES (?, ?)', (sentence, embedding))
                conn.commit()
            except sqlite3.IntegrityError:
                print(f"⚠️ Sentence '{sentence}' already exists in the database.")

    def compute_embeddings(self, sent_list):
        """Compute and normalize embeddings for a list of sentences."""
        encoded_input = self.tokenizer(sent_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return self.normalize_embeddings(embeddings)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """Apply mean pooling to the model output."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def normalize_embeddings(embeddings):
        """Normalize the embeddings to unit vectors."""
        return F.normalize(embeddings, p=2, dim=1)

    def add_sentence(self, sentence):
        """Add a new sentence to the database and embeddings."""
        embedding = self.compute_embeddings([sentence]).detach()
        embedding_np = embedding.numpy().astype('float32')
        embedding_bytes = pickle.dumps(embedding_np)
        self.save_to_db(sentence, embedding_bytes)

        # Update in-memory cache
        self.sentences.append(sentence)
        if self.sentence_embeddings.nelement() == 0:
            self.sentence_embeddings = embedding
        else:
            self.sentence_embeddings = torch.cat((self.sentence_embeddings, embedding), dim=0)

        print(f"Added: '{sentence}'")

    def remove_sentence(self, sentence):
        """Remove a sentence from the database and embeddings."""
        sentence = sentence.strip()  # Normalize input sentence
        
        if sentence in self.sentences:
            idx = self.sentences.index(sentence)
            self.sentences.pop(idx)
            if self.sentence_embeddings.nelement() > 0:
                self.sentence_embeddings = torch.cat(
                    [self.sentence_embeddings[:idx], self.sentence_embeddings[idx + 1:]], dim=0
                )
            self.remove_from_db(sentence)  # Remove from DB using original sentence
            print(f"Removed: '{sentence}'")
        else:
            print("⚠️ Sentence not found.")

    def remove_from_db(self, sentence):
        """Remove a sentence from the database."""
        with sqlite3.connect('semantic_search.db') as conn:
            c = conn.cursor()
            c.execute('DELETE FROM sentences WHERE text = ?', (sentence,))
            conn.commit()

    def search(self, query, threshold=0.5):
        """Search for similar sentences based on a query."""
        query_embedding = self.compute_embeddings([query]).detach()
        similarities = F.cosine_similarity(query_embedding, self.sentence_embeddings)

        top_indices = (similarities >= threshold).nonzero(as_tuple=True)[0].tolist()

        if top_indices:
            top_k = min(3, len(top_indices))
            top_scores, sorted_indices = similarities[top_indices].topk(top_k)

            print("\nTop matches:")
            for rank, (idx, score) in enumerate(zip([top_indices[i] for i in sorted_indices], top_scores), start=1):
                print(f"{rank}. '{self.sentences[idx]}' (similarity score: {score:.4f})")
        else:
            print("⚠️ No matches found above the specified threshold.")

    def view_sentences(self):
        """Display all current sentences."""
        if not self.sentences:
            print("⚠️ No sentences found.")
        else:
            print("\nCurrent sentences:")
            for idx, sentence in enumerate(self.sentences, start=1):
                print(f"{idx}. '{sentence}'")

    def clear_all_sentences(self):
        """Clear all sentences from the in-memory list and the database."""
        self.sentences.clear()
        self.sentence_embeddings = torch.empty((0,))
        with sqlite3.connect('semantic_search.db') as conn:
            c = conn.cursor()
            c.execute('DELETE FROM sentences')
            conn.commit()
        print("All sentences cleared.")

    def run(self):
        """Run the interactive command loop."""
        print("Semantic Search System Ready!")
        print("Commands: 'add', 'remove', 'search', 'view', 'clear', 'exit'")

        while True:
            command = input("\nEnter a command: ").strip().lower()
            if command == "exit":
                # Clear sentences and embeddings before exiting
                self.sentences.clear()
                self.sentence_embeddings = torch.empty((0,))
                print("Exiting semantic search. Goodbye!")
                break
            elif command == "add":
                new_sentence = input("Enter a new sentence: ").strip()
                if new_sentence:
                    self.add_sentence(new_sentence)
                else:
                    print("⚠️ Empty sentence not added.")
            elif command == "remove":
                sentence_to_remove = input("Enter the sentence to remove: ").strip()
                self.remove_sentence(sentence_to_remove)
            elif command == "search":
                query = input("Enter a sentence to search: ").strip()
                if query:
                    threshold_input = input("Enter a similarity threshold (0 to 1): ")
                    try:
                        threshold = float(threshold_input)
                        if not (0 <= threshold <= 1):
                            raise ValueError
                    except ValueError:
                        print("⚠️ Invalid threshold. Please enter a number between 0 and 1.")
                        continue
                    self.search(query, threshold)
                else:
                    print("⚠️ Please enter a valid query.")
            elif command == "view":
                self.view_sentences()  # View all current sentences
            elif command == "clear":
                self.clear_all_sentences()  # Clear all sentences
            else:
                print("⚠️ Unknown command. Please try again.")

if __name__ == "__main__":
    search_system = SemanticSearch()
    search_system.run()
