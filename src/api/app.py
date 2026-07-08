"""FastAPI service exposing the triage agent.

`POST /triage` runs a case through the model returned by `src.agent.model.get_model`
(the deterministic stub by default), so the service works with no LLM API key.
Case memory is a single process-wide instance — fine for this demo service.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.decision import TriageDecision
from src.agent.model import get_model
from src.agent.run import run_triage
from src.memory.case_memory import CaseMemory

app = FastAPI(title="KYC Fraud Triage Agent")
_memory = CaseMemory()


class CaseRequest(BaseModel):
    """A customer case to triage: identity attributes plus transaction context."""

    entity_id: str
    full_name: str
    date_of_birth: str
    transaction: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageDecision)
def triage(case: CaseRequest) -> TriageDecision:
    return run_triage(case.model_dump(), get_model(), _memory)
