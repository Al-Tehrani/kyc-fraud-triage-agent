from src.rag.chunking import Chunk
from src.rag.index import PolicyIndex


def test_search_returns_the_most_semantically_similar_chunk():
    chunks = [
        Chunk(text="Cats are small furry pets that like to nap in sunlight.", source="pets"),
        Chunk(text="A confirmed sanctions list match requires mandatory escalation.", source="sanctions"),
        Chunk(text="The quarterly financial report is due at the end of the fiscal year.", source="finance"),
    ]
    index = PolicyIndex.build(chunks)

    results = index.search("What happens when a customer matches the sanctions list?", k=1)

    assert len(results) == 1
    assert results[0].source == "sanctions"


def test_search_respects_k():
    chunks = [Chunk(text=f"Policy text number {i}.", source=f"doc_{i}") for i in range(5)]
    index = PolicyIndex.build(chunks)

    results = index.search("policy text", k=2)

    assert len(results) == 2
