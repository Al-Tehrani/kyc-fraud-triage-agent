from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import build_graph


class FakeEchoModel:
    """Deterministic stand-in for a chat model: no live LLM calls needed.

    Calls the `echo` tool once, then answers using the tool's result — enough
    to exercise a full reason -> act -> observe -> decide cycle.
    """

    def invoke(self, messages: list) -> AIMessage:
        last = messages[-1]
        if last.type == "tool":
            return AIMessage(content=f"Final answer: {last.content}")
        return AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": last.content}, "id": "call_1"}],
        )


def test_react_loop_calls_tool_then_answers():
    graph = build_graph(FakeEchoModel())

    result = graph.invoke({"messages": [HumanMessage(content="hello")]})

    messages = result["messages"]
    assert messages[0].content == "hello"
    assert messages[1].tool_calls[0]["name"] == "echo"
    assert messages[2].type == "tool"
    assert messages[2].content == "hello"
    assert messages[3].content == "Final answer: hello"
