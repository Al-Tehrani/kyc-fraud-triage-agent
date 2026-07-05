"""The `retrieve_policy` tool: semantic search over the KYC/AML policy docs."""

from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from src.rag.chunking import chunk_policy_docs
from src.rag.index import PolicyIndex

POLICIES_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"


@lru_cache(maxsize=1)
def _get_index() -> PolicyIndex:
    """Build the policy index once per process and reuse it on every call."""
    return PolicyIndex.build(chunk_policy_docs(POLICIES_DIR))


@tool
def retrieve_policy(query: str, k: int = 3) -> list[dict[str, str]]:
    """Semantically search KYC/AML policy docs and return the top-k matching chunks.

    Each result includes the chunk text and its source document name, so the
    agent can cite the policy backing its decision.
    """
    return [{"source": chunk.source, "text": chunk.text} for chunk in _get_index().search(query, k=k)]
