from langchain_core.tools import tool


@tool
def echo(text: str) -> str:
    """Return the input text unchanged. Dummy tool for exercising the ReAct loop."""
    return text
