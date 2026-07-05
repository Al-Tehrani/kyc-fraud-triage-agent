"""Local sentence embeddings, no API calls."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process (weights cache on disk after first download)."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts as L2-normalized vectors, so inner product == cosine similarity."""
    return get_embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
