"""A FAISS index over policy chunks, searchable by embedding similarity."""

import faiss

from src.rag.chunking import Chunk
from src.rag.embeddings import embed


class PolicyIndex:
    def __init__(self, index: faiss.Index, chunks: list[Chunk]):
        self._index = index
        self._chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "PolicyIndex":
        vectors = embed([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors = cosine similarity
        index.add(vectors)
        return cls(index, chunks)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        query_vector = embed([query])
        _, neighbor_indices = self._index.search(query_vector, k)
        return [self._chunks[i] for i in neighbor_indices[0] if i != -1]
