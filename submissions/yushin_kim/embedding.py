import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
enc = None

def get_encoder():
    """
    Loads and returns the sentence transformer model.
    Initializes the model only once.
    """
    global enc
    if enc is None:
        enc = SentenceTransformer(MODEL_ID)
    return enc

def embed(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of texts into normalized vectors.
    """
    vecs = get_encoder().encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def centroid(tags: list[str]) -> np.ndarray:
    """
    Calculates the centroid vector for a list of tags.
    """
    V = embed(tags)
    c = V.mean(axis=0)
    n = np.linalg.norm(c) + 1e-9
    return c / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    return float(np.dot(a, b))
