import json

from langchain_core.messages import AIMessage

from src.agent.run import run_triage
from src.memory.case_memory import CaseMemory

CASE = {"entity_id": "cust_001", "full_name": "Jordan Smith", "date_of_birth": "1990-05-02"}


class ImmediateDecisionModel:
    """Answers with a fixed decision immediately, no tool calls. Exercises the
    run_triage plumbing (memory lookup, prompt building, memory write) in
    isolation from the tool-calling loop, which test_graph.py covers."""

    def __init__(self, decision: dict):
        self._decision_json = json.dumps(decision)
        self.seen_messages: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.seen_messages.append(messages)
        return AIMessage(content=self._decision_json)


def test_run_triage_returns_decision_and_records_it_in_memory():
    memory = CaseMemory()
    model = ImmediateDecisionModel(
        {"decision": "approve", "confidence": 0.9, "cited_policies": ["risk_score_thresholds"], "reasoning": "ok"}
    )

    decision = run_triage(CASE, model, memory)

    assert decision.decision == "approve"
    assert memory.get("cust_001").decision == decision


def test_run_triage_injects_prior_context_for_repeat_entity():
    memory = CaseMemory()
    model = ImmediateDecisionModel(
        {"decision": "review", "confidence": 0.5, "cited_policies": [], "reasoning": "gray zone"}
    )

    run_triage(CASE, model, memory)
    run_triage(CASE, model, memory)

    first_prompt = model.seen_messages[0][-1].content
    second_prompt = model.seen_messages[1][-1].content
    assert "prior case" not in first_prompt.lower()
    assert "prior case on file: decision=review" in second_prompt.lower()


def test_run_triage_captures_fraud_and_sanctions_results_from_tool_trace():
    memory = CaseMemory()

    class ToolCallingModel:
        def __init__(self):
            self._step = 0

        def invoke(self, messages: list) -> AIMessage:
            self._step += 1
            if self._step == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "fraud_score",
                            "args": {"transaction": {f"v{i}": 0.0 for i in range(1, 29)} | {"amount": 20.0}},
                            "id": "call_1",
                        }
                    ],
                )
            if self._step == 2:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "sanctions_check", "args": {"name": "Jordan Smith", "dob": "1990-05-02"}, "id": "call_2"}
                    ],
                )
            return AIMessage(
                content=json.dumps(
                    {"decision": "approve", "confidence": 0.9, "cited_policies": [], "reasoning": "ok"}
                )
            )

    run_triage(CASE, ToolCallingModel(), memory)

    record = memory.get("cust_001")
    assert record.fraud_probability is not None
    assert record.sanctions_match_status == "no_hit"
