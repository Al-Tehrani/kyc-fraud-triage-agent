"""Deterministic stand-in for a live LLM chat model.

Runs the same evidence-gathering sequence a real model would (retrieve
policy, score fraud, screen sanctions) and applies the decision rule from
`src.agent.prompts.SYSTEM_PROMPT`. Lets the agent graph run end-to-end with
no LLM API key. It implements the same `ChatModel` protocol (`src.agent.graph`)
that a real LangChain chat model bound to TOOLS would, so `src.agent.model`
can swap one for the other without touching the graph.
"""

import json

from langchain_core.messages import AIMessage, BaseMessage

from src.agent.decision import TriageDecision


def _extract_case(messages: list[BaseMessage]) -> dict:
    """Pull the case JSON back out of the human prompt built by `src.agent.run`."""
    for message in messages:
        if message.type == "human":
            _, case_line, *_ = message.content.split("\n", 2)
            return json.loads(case_line)
    raise ValueError("no human message containing a case payload found")


def _tool_result(messages: list[BaseMessage], name: str) -> dict | list | None:
    for message in reversed(messages):
        if message.type == "tool" and message.name == name:
            return json.loads(message.content)
    return None


def _decide(fraud: dict, sanctions: dict, cited_policies: list[str]) -> TriageDecision:
    """Apply the documented decision rule (risk_score_thresholds.md,
    escalation_criteria.md, sanctions_screening.md):

    - a confirmed sanctions match always escalates, regardless of fraud score.
    - otherwise, fraud probability against the policy thresholds decides:
      below 0.30 -> approve, 0.30-0.70 -> review, above 0.70 -> escalate.
    - a sanctions candidate false positive (name match, DOB differs) must
      never be auto-cleared, so it forces at least a review.

    Confidence reflects how clear-cut the case is: a confirmed match is a
    hard rule (fixed high confidence); an approve/escalate driven by fraud
    score is more confident the further the probability sits from its
    threshold; a review is inherently the least clear-cut band.
    """
    probability = fraud["fraud_probability"]
    risk_band = fraud["risk_band"]
    match_status = sanctions["match_status"]

    if match_status == "confirmed_match":
        decision, confidence = "escalate", 0.95
        reasoning = (
            f"Sanctions screening returned a confirmed match, which forces escalation "
            f"regardless of fraud score (fraud probability {probability}, {risk_band} risk)."
        )
    elif risk_band == "high":
        decision, confidence = "escalate", probability
        reasoning = (
            f"Fraud probability {probability} is above the 0.70 escalate threshold "
            f"({risk_band} risk); sanctions screening returned {match_status}."
        )
    elif risk_band == "low" and match_status == "no_hit":
        decision, confidence = "approve", 1 - probability
        reasoning = (
            f"Fraud probability {probability} is below the 0.30 approve threshold "
            f"({risk_band} risk) and sanctions screening returned {match_status}."
        )
    else:
        decision = "review"
        confidence = 0.5 + abs(probability - 0.5) if risk_band == "gray_zone" else 0.6
        reasoning = (
            f"Fraud probability {probability} ({risk_band} risk) and sanctions status "
            f"{match_status} don't clear the approve bar or force escalation, so this "
            "needs analyst review."
        )

    return TriageDecision(
        decision=decision,
        confidence=round(confidence, 2),
        cited_policies=cited_policies,
        reasoning=reasoning,
    )


class StubTriageModel:
    """Runs one fixed tool sequence per case, then emits a structured decision.

    One instance handles exactly one case (it tracks its own step count), so
    callers must get a fresh instance per triage run.
    """

    def __init__(self) -> None:
        self._step = 0
        self._case: dict | None = None

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self._step += 1
        if self._case is None:
            self._case = _extract_case(messages)

        if self._step == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_policy",
                        "args": {"query": "risk score thresholds and escalation criteria", "k": 2},
                        "id": "call_1",
                    }
                ],
            )
        if self._step == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "fraud_score", "args": {"transaction": self._case["transaction"]}, "id": "call_2"}
                ],
            )
        if self._step == 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sanctions_check",
                        "args": {"name": self._case["full_name"], "dob": self._case["date_of_birth"]},
                        "id": "call_3",
                    }
                ],
            )

        policy_chunks = _tool_result(messages, "retrieve_policy") or []
        fraud = _tool_result(messages, "fraud_score")
        sanctions = _tool_result(messages, "sanctions_check")
        cited_policies = sorted({chunk["source"] for chunk in policy_chunks})
        return AIMessage(content=_decide(fraud, sanctions, cited_policies).model_dump_json())
