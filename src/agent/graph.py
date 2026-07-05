"""Minimal ReAct-style loop: agent reasons, optionally calls a tool, observes, repeats.

    (agent) --tool call requested--> (tools) --tool result--> (agent) --> END
       \\_______________________ no tool call requested ______________________/
"""

from typing import Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import echo

TOOLS = [echo]


class ChatModel(Protocol):
    """Anything with an `invoke(messages) -> AIMessage` method, e.g. a LangChain
    chat model with TOOLS already bound, or a stub used in tests."""

    def invoke(self, messages: list) -> AIMessage: ...


def _call_model(state: AgentState, model: ChatModel) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_graph(model: ChatModel):
    """Wire the agent <-> tools loop. `model` must already have TOOLS bound."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: _call_model(state, model))
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
