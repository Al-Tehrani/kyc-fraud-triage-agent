from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Graph state: a running list of chat messages (human, AI, tool)."""

    messages: Annotated[list[BaseMessage], add_messages]
