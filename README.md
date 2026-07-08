# KYC Fraud Triage Agent

[![CI](https://github.com/Al-Tehrani/kyc-fraud-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Al-Tehrani/kyc-fraud-triage-agent/actions/workflows/ci.yml)

An agentic AI system for identity verification and fraud prevention. Given a
customer case (identity attributes + transaction context), a LangGraph
ReAct-style agent gathers evidence via tools (policy retrieval, fraud
scoring, sanctions screening) and produces an auditable triage decision:
**approve**, **review**, or **escalate**.

See [CLAUDE.md](CLAUDE.md) for architecture, stack, and build milestones.

Full write-up (architecture diagram, metrics, motivation) lands in a later
milestone.

## Demo

A Gradio demo (`src/demo/app.py`) runs the agent on the stub model — no API
key needed. Try it locally:

```
python app.py
```

Then open the printed local URL and either click one of the three preloaded
cases (clear approve, gray-zone review, clear escalate) or edit the case JSON
and click "Run triage".
