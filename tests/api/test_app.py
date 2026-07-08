from fastapi.testclient import TestClient

from src.api.app import app
from src.ml.data import FEATURE_NAMES

client = TestClient(app)


def _legit_transaction() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES[:-1]} | {"amount": 20.0}


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_triage_approves_clean_case():
    case = {
        "entity_id": "cust_100",
        "full_name": "Jordan Smith",
        "date_of_birth": "2000-01-01",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "approve"
    assert body["cited_policies"]


def test_triage_escalates_on_confirmed_sanctions_match():
    case = {
        "entity_id": "cust_101",
        "full_name": "Anwar Al-Masri",
        "date_of_birth": "1975-03-14",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    assert response.json()["decision"] == "escalate"


def test_triage_reviews_candidate_false_positive_sanctions_match():
    case = {
        "entity_id": "cust_102",
        "full_name": "Anwar Al-Masri",
        "date_of_birth": "1990-01-01",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    assert response.json()["decision"] == "review"


def test_triage_response_matches_decision_schema():
    case = {
        "entity_id": "cust_103",
        "full_name": "Jordan Smith",
        "date_of_birth": "2000-01-01",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    assert set(response.json().keys()) == {"decision", "confidence", "cited_policies", "reasoning"}
