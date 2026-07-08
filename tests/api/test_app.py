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


def test_triage_rejects_transaction_missing_required_feature():
    transaction = _legit_transaction()
    del transaction["v14"]
    case = {
        "entity_id": "cust_104",
        "full_name": "Jordan Smith",
        "date_of_birth": "2000-01-01",
        "transaction": transaction,
    }

    response = client.post("/triage", json=case)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "v14" and error["type"] == "missing" for error in errors)


def test_triage_rejects_transaction_with_non_numeric_feature():
    transaction = _legit_transaction() | {"amount": "not-a-number"}
    case = {
        "entity_id": "cust_105",
        "full_name": "Jordan Smith",
        "date_of_birth": "2000-01-01",
        "transaction": transaction,
    }

    response = client.post("/triage", json=case)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "amount" for error in errors)
