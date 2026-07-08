from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.agent.decision import TriageDecision


class AgentState(TypedDict):
    """Graph state: a running list of chat messages, plus the final decision
    once the agent has finished gathering evidence."""

    messages: Annotated[list[BaseMessage], add_messages]
    decision: TriageDecision | None
