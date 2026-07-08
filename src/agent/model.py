"""Selects the chat model backing the triage agent.

Defaults to the deterministic `StubTriageModel` so the graph runs end-to-end
with no LLM API key. Live mode is gated behind the `TRIAGE_LIVE` env var;
wiring a real LangChain chat model bound to TOOLS is a later milestone. This
factory is the seam the API (and anything else) should call through, rather
than importing a specific model class directly.
"""

import os

from src.agent.graph import ChatModel
from src.agent.stub_model import StubTriageModel


def get_model() -> ChatModel:
    """Return a fresh model instance for one triage run."""
    if os.environ.get("TRIAGE_LIVE") == "1":
        raise NotImplementedError(
            "Live LLM mode is not wired up yet. Unset TRIAGE_LIVE (or set it to 0) to use the stub model."
        )
    return StubTriageModel()
