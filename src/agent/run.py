"""Runs one case through the triage graph and persists the outcome to memory."""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agent.decision import TriageDecision
from src.agent.graph import ChatModel, build_graph
from src.agent.prompts import SYSTEM_PROMPT
from src.memory.case_memory import CaseMemory, CaseRecord


def _build_messages(case: dict, prior: CaseRecord | None) -> list[BaseMessage]:
    content = f"Triage this case:\n{json.dumps(case)}"
    if prior is not None:
        content += (
            f"\n\nThis entity has a prior case on file: decision={prior.decision.decision}, "
            f"fraud_probability={prior.fraud_probability}, "
            f"sanctions_match_status={prior.sanctions_match_status}."
        )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]


def _last_tool_result(messages: list[BaseMessage], tool_name: str) -> dict | None:
    """Find the most recent result of `tool_name` in the message trace, if any."""
    for message in reversed(messages):
        if message.type == "tool" and message.name == tool_name:
            return json.loads(message.content)
    return None


def run_triage(case: dict, model: ChatModel, memory: CaseMemory) -> TriageDecision:
    """Triage `case`, using and updating `memory` for the case's entity_id."""
    entity_id = case["entity_id"]
    prior = memory.get(entity_id)

    graph = build_graph(model)
    result = graph.invoke({"messages": _build_messages(case, prior)})
    decision: TriageDecision = result["decision"]

    fraud_result = _last_tool_result(result["messages"], "fraud_score")
    sanctions_result = _last_tool_result(result["messages"], "sanctions_check")
    memory.record(
        CaseRecord(
            entity_id=entity_id,
            decision=decision,
            fraud_probability=fraud_result["fraud_probability"] if fraud_result else None,
            sanctions_match_status=sanctions_result["match_status"] if sanctions_result else None,
        )
    )
    return decision
