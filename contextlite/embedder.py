import numpy as np
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2: 80MB, CPU-friendly, ~10ms/sentence, good semantic quality
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns L2-normalized vectors (dot product = cosine sim)."""
    return get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def batch_cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Score one query vector against a matrix of embeddings. Returns 1D array."""
    return matrix @ query_vec
