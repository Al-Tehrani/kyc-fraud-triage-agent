"""Gradio demo for the KYC fraud triage agent.

Runs entirely on the deterministic stub model (`src.agent.model.get_model`
with no `TRIAGE_LIVE` env var set), so it needs no LLM API key — a visitor
can load one of three preloaded cases or edit the JSON and triage it live.

Entry point for Hugging Face Spaces: the root `app.py` imports `demo` from
here. See that file for the Spaces deployment settings.
"""

import json

import gradio as gr

from src.agent.model import get_model
from src.agent.run import run_triage
from src.demo.examples import EXAMPLES
from src.memory.case_memory import CaseMemory

_memory = CaseMemory()
_DECISION_EMOJI = {"approve": "✅", "review": "\U0001f7e1", "escalate": "\U0001f6a8"}


def _format_decision(entity_id: str, decision_json: str) -> str:
    decision = json.loads(decision_json)
    prior = _memory.get(entity_id)
    lines = [
        f"## {_DECISION_EMOJI[decision['decision']]} {decision['decision'].upper()}"
        f" (confidence {decision['confidence']:.0%})",
        "",
        "**Reasoning:** " + decision["reasoning"],
        "",
        "**Cited policies:** " + (", ".join(decision["cited_policies"]) or "none"),
    ]
    if prior is not None:
        lines += [
            "",
            f"**Fraud probability:** {prior.fraud_probability}  \n"
            f"**Sanctions match:** {prior.sanctions_match_status}",
        ]
    return "\n".join(lines)


def _run_case(case_json: str) -> tuple[str, str]:
    """Triage the case in `case_json`. Returns (readable summary, raw decision JSON)."""
    try:
        case = json.loads(case_json)
    except json.JSONDecodeError as exc:
        return f"**Invalid JSON:** {exc}", ""

    try:
        decision = run_triage(case, get_model(), _memory)
    except Exception as exc:  # UI boundary: show the error instead of crashing the app
        return f"**Could not triage this case:** {exc}", ""

    decision_json = decision.model_dump_json()
    return _format_decision(case["entity_id"], decision_json), decision_json


def _load_example(name: str) -> tuple[str, str, str]:
    case_json = json.dumps(EXAMPLES[name], indent=2)
    summary, raw = _run_case(case_json)
    return case_json, summary, raw


with gr.Blocks(title="KYC Fraud Triage Agent") as demo:
    gr.Markdown(
        "# KYC Fraud Triage Agent\n"
        "Agentic triage over identity + transaction data: policy retrieval, "
        "fraud scoring, and sanctions screening feed a structured, auditable "
        "decision. Runs on a deterministic stub model, so no API key is needed."
    )

    with gr.Row():
        example_buttons = [gr.Button(name) for name in EXAMPLES]

    case_input = gr.Code(
        value=json.dumps(next(iter(EXAMPLES.values())), indent=2),
        language="json",
        label="Case JSON (edit freely, or load an example above)",
        lines=16,
    )
    run_button = gr.Button("Run triage", variant="primary")

    summary_output = gr.Markdown(label="Decision")
    raw_output = gr.Code(language="json", label="Raw TriageDecision")

    for name, button in zip(EXAMPLES, example_buttons):
        button.click(fn=lambda n=name: _load_example(n), outputs=[case_input, summary_output, raw_output])

    run_button.click(fn=_run_case, inputs=case_input, outputs=[summary_output, raw_output])


if __name__ == "__main__":
    demo.launch()
