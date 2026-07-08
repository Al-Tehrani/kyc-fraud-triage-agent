"""FastAPI service exposing the triage agent.

`POST /triage` runs a case through the model returned by `src.agent.model.get_model`
(the deterministic stub by default), so the service works with no LLM API key.
Case memory is a single process-wide instance — fine for this demo service.
"""

from fastapi import FastAPI
from pydantic import BaseModel, create_model

from src.agent.decision import TriageDecision
from src.agent.model import get_model
from src.agent.run import run_triage
from src.memory.case_memory import CaseMemory
from src.ml.data import FEATURE_NAMES

app = FastAPI(title="KYC Fraud Triage Agent")
_memory = CaseMemory()

# Built from FEATURE_NAMES (v1..v28, amount) rather than listed by hand, so it
# can't drift from what src.tools.fraud_score actually requires. Every field
# is required and must be numeric, so a missing or malformed feature fails
# request validation with a 422 instead of a KeyError/500 inside the tool.
TransactionFeatures = create_model(
    "TransactionFeatures", **{name: (float, ...) for name in FEATURE_NAMES}
)


class CaseRequest(BaseModel):
    """A customer case to triage: identity attributes plus transaction context."""

    entity_id: str
    full_name: str
    date_of_birth: str
    transaction: TransactionFeatures


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "KYC Fraud Triage Agent API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageDecision)
def triage(case: CaseRequest) -> TriageDecision:
    return run_triage(case.model_dump(), get_model(), _memory)
