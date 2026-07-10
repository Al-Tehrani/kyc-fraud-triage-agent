# KYC Fraud Triage Agent

## What this project is
An agentic AI system for identity verification and fraud prevention, inspired by real KYC/AML compliance workflows. The agent receives a customer case (JSON with identity attributes and transaction context) and produces a triage decision: **approve**, **review**, or **escalate**, with cited policy references and step-by-step reasoning.

Purpose: portfolio project demonstrating agent orchestration, RAG, tool use, ML integration, and production engineering practices for an ML Engineer role focused on agentic systems in fraud/KYC.

## Architecture
1. **Agent orchestration:** LangGraph state graph implementing a ReAct-style loop (reason, act via tools, observe, decide).
2. **RAG layer:** FAISS vector store over synthetic KYC/AML policy documents in `data/policies/` (markdown). Exposed to the agent as a `retrieve_policy` tool. Sentence-transformers for embeddings (local, no API cost).
3. **Tools:**
   - `retrieve_policy(query)`: semantic search over policy docs, returns relevant policy chunks with source references.
   - `fraud_score(case)`: XGBoost classifier trained on the Kaggle credit card fraud dataset, returns probability and top contributing features.
   - `sanctions_check(name, dob)`: mock watchlist lookup against a small synthetic sanctions/PEP list, returns hit/no-hit with match details.
4. **Memory:** simple case memory keyed by entity ID so repeat entities carry prior context.
5. **Decision logic:** two-threshold triage. Low fraud score + no hits = approve. High score or sanctions hit = escalate. Middle band = review. The agent must output structured JSON: decision, confidence, cited_policies, reasoning.
6. **API:** FastAPI service exposing `POST /triage` and `GET /health`.
7. **Demo mode:** three cached example cases (clear approve, gray-zone review, clear escalate) that run without live LLM calls, plus live mode gated behind an API key env var.

## Stack and environment
- Python 3.11+, running in WSL2 Ubuntu
- Dedicated venv at `.venv/` (NOT the conda base environment; do not install into conda)
- Core deps: langgraph, langchain, faiss-cpu, sentence-transformers, xgboost, scikit-learn, pandas, fastapi, uvicorn, pydantic, pytest, httpx
- Secrets via environment variables / `.env` (never commit; `.env` is gitignored)

## Project structure
```
src/
  agent/        # LangGraph graph, state, nodes
  tools/        # retrieve_policy, fraud_score, sanctions_check
  rag/          # ingestion, chunking, FAISS index build/load
  ml/           # XGBoost training script and saved model
  api/          # FastAPI app
  memory/       # case memory
data/
  policies/     # synthetic KYC/AML policy markdown docs
  sanctions/    # synthetic watchlist CSV
tests/          # pytest suites mirroring src/
```

## Conventions
- Clean, typed, testable Python. Type hints everywhere, docstrings on public functions.
- Every feature lands with at least one pytest test. Run `pytest -q` before any commit.
- Small, focused commits with descriptive messages after each working milestone.
- No secrets, API keys, or large binaries in git. Model artifacts go in `src/ml/artifacts/` (gitignored) with a training script to reproduce.
- Keep LLM prompts in dedicated files/constants, not scattered inline.
- Prefer deterministic, mockable tool interfaces so tests never require live LLM calls.

## Build milestones (work on ONE at a time, wait for confirmation)
1. Scaffold structure, venv, pyproject.toml/requirements, git init, this file committed.
2. Minimal LangGraph agent with ReAct loop calling one dummy tool, with a passing end-to-end test.
3. RAG: policy docs, FAISS ingestion, retrieve_policy tool, tests.
4. fraud_score tool: train XGBoost on Kaggle creditcard.csv, save artifact, wrap as tool, tests. sanctions_check mock tool, tests.
5. Memory + structured triage decision output, tests.
6. FastAPI /triage endpoint, tests via httpx.
7. Dockerfile + GitHub Actions CI (pytest on push).
8. Harden /triage input validation; deploy the FastAPI service to Render.
9. README polish: architecture diagram (Mermaid), metrics discussion, "why this matters for identity verification" section.

## Owner
Ali (AL Tehrani), ML engineer. MMath Data Science (Waterloo), thesis on Interpretable ML. The decision output design intentionally reflects explainability principles: every triage decision must be auditable, with cited policy and reasoning a compliance analyst could follow.
