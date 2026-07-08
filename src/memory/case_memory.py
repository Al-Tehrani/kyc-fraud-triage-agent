"""Simple in-process case memory keyed by entity ID.

Lets a repeat entity's prior triage outcome carry into a new case instead of
starting from scratch each time. Not persisted across process restarts —
that's a concern for a later milestone if needed.
"""

from dataclasses import dataclass

from src.agent.decision import TriageDecision


@dataclass
class CaseRecord:
    entity_id: str
    decision: TriageDecision
    fraud_probability: float | None = None
    sanctions_match_status: str | None = None


class CaseMemory:
    """Keyed by entity_id; each new case overwrites that entity's prior record."""

    def __init__(self) -> None:
        self._records: dict[str, CaseRecord] = {}

    def get(self, entity_id: str) -> CaseRecord | None:
        return self._records.get(entity_id)

    def record(self, case_record: CaseRecord) -> None:
        self._records[case_record.entity_id] = case_record
