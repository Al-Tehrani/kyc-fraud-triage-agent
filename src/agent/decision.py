"""The structured triage decision the agent must produce for every case."""

from typing import Literal

from pydantic import BaseModel, Field


class TriageDecision(BaseModel):
    """Auditable output: a compliance analyst must be able to follow the reasoning."""

    decision: Literal["approve", "review", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)
    cited_policies: list[str]
    reasoning: str
