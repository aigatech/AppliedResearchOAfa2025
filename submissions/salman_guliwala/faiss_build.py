import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_faiss_index(corpus_file="corpus.txt", index_file="faiss.index"):
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    build_faiss_index()
