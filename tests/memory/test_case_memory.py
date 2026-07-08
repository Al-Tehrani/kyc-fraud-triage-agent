from src.agent.decision import TriageDecision
from src.memory.case_memory import CaseMemory, CaseRecord


def _decision(**overrides) -> TriageDecision:
    payload = {
        "decision": "approve",
        "confidence": 0.9,
        "cited_policies": ["risk_score_thresholds"],
        "reasoning": "Low fraud score, no sanctions hit.",
    }
    return TriageDecision.model_validate(payload | overrides)


def test_unknown_entity_returns_none():
    memory = CaseMemory()

    assert memory.get("cust_unknown") is None


def test_record_then_get_round_trips():
    memory = CaseMemory()
    record = CaseRecord(entity_id="cust_001", decision=_decision(), fraud_probability=0.12)

    memory.record(record)

    assert memory.get("cust_001") == record


def test_recording_again_overwrites_prior_record_for_same_entity():
    memory = CaseMemory()
    memory.record(CaseRecord(entity_id="cust_001", decision=_decision(decision="approve")))
    memory.record(CaseRecord(entity_id="cust_001", decision=_decision(decision="escalate")))

    assert memory.get("cust_001").decision.decision == "escalate"
