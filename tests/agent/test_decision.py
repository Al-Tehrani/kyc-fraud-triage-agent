import pytest
from pydantic import ValidationError

from src.agent.decision import TriageDecision


def test_valid_decision_parses():
    decision = TriageDecision.model_validate_json(
        '{"decision": "review", "confidence": 0.55, '
        '"cited_policies": ["risk_score_thresholds"], "reasoning": "Gray-zone fraud score."}'
    )

    assert decision.decision == "review"
    assert decision.confidence == 0.55


@pytest.mark.parametrize(
    "field, value",
    [("decision", "maybe"), ("confidence", 1.5), ("confidence", -0.1)],
)
def test_invalid_values_are_rejected(field, value):
    payload = {
        "decision": "approve",
        "confidence": 0.9,
        "cited_policies": [],
        "reasoning": "x",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        TriageDecision.model_validate(payload)
