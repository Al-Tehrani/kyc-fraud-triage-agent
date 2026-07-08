import json

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.decision import TriageDecision
from src.agent.graph import build_graph

SAMPLE_TRANSACTION = {f"v{i}": 0.0 for i in range(1, 29)} | {"amount": 50.0}


class StubTriageModel:
    """Deterministic stand-in for an LLM: retrieves a policy, scores fraud,
    screens sanctions, then emits a structured decision. Exercises a full
    reason -> act -> observe -> decide cycle with no live LLM calls."""

    def __init__(self):
        self._step = 0

    def invoke(self, messages: list) -> AIMessage:
        self._step += 1
        if self._step == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_policy", "args": {"query": "fraud score thresholds", "k": 1}, "id": "call_1"}
                ],
            )
        if self._step == 2:
            return AIMessage(
                content="",
                tool_calls=[{"name": "fraud_score", "args": {"transaction": SAMPLE_TRANSACTION}, "id": "call_2"}],
            )
        if self._step == 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "sanctions_check", "args": {"name": "Jordan Smith", "dob": "1990-05-02"}, "id": "call_3"}
                ],
            )
        return AIMessage(
            content=json.dumps(
                {
                    "decision": "approve",
                    "confidence": 0.92,
                    "cited_policies": ["risk_score_thresholds"],
                    "reasoning": "Fraud probability is low and no sanctions hit was found.",
                }
            )
        )


def test_full_triage_flow_produces_structured_decision():
    graph = build_graph(StubTriageModel())

    result = graph.invoke({"messages": [HumanMessage(content="Triage case for Jordan Smith")]})

    decision = result["decision"]
    assert isinstance(decision, TriageDecision)
    assert decision.decision == "approve"
    assert decision.cited_policies == ["risk_score_thresholds"]

    tool_calls_made = [m.name for m in result["messages"] if m.type == "tool"]
    assert tool_calls_made == ["retrieve_policy", "fraud_score", "sanctions_check"]
