"""ReAct-style triage loop: agent reasons, optionally calls a tool, observes,
repeats. Once it stops calling tools, its final answer is parsed into a
structured TriageDecision.

    (agent) --tool call requested--> (tools) --tool result--> (agent)
       \\_______________________ no tool call requested ______________________/
                                        |
                                        v
                                   (finalize) --> END
"""

from typing import Protocol

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.decision import TriageDecision
from src.agent.state import AgentState
from src.tools.fraud_score import fraud_score
from src.tools.retrieve_policy import retrieve_policy
from src.tools.sanctions_check import sanctions_check

TOOLS = [retrieve_policy, fraud_score, sanctions_check]


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
    return "finalize"


def _finalize_decision(state: AgentState) -> dict:
    """Parse the model's final answer into the structured triage decision."""
    last_message = state["messages"][-1]
    return {"decision": TriageDecision.model_validate_json(last_message.content)}


def build_graph(model: ChatModel):
    """Wire the agent <-> tools loop. `model` must already have TOOLS bound."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: _call_model(state, model))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("finalize", _finalize_decision)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()
