"""A TF-IDF index over policy chunks, searchable by cosine similarity.

Swapped from sentence-transformers + FAISS: for a five-document policy
corpus, a neural embedding model is massive overkill and was the largest RAM
consumer on our 512MB Render instance. TfidfVectorizer needs no model
weights and is plenty for keyword-level matching over a corpus this small.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.rag.chunking import Chunk


class PolicyIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: list[Chunk]):
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "PolicyIndex":
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(vectorizer, matrix, chunks)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        query_vector = self._vectorizer.transform([query])
        # TF-IDF rows are L2-normalized by default, so the linear kernel
        # (plain dot product) is equivalent to cosine similarity.
        scores = linear_kernel(query_vector, self._matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [self._chunks[i] for i in ranked if scores[i] > 0]
