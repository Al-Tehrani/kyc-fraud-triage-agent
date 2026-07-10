from fastapi.testclient import TestClient

from src.api.app import app
from src.ml.data import FEATURE_NAMES

client = TestClient(app)


def _transaction(v_scale: float, amount: float) -> dict[str, float]:
    return {name: v_scale for name in FEATURE_NAMES[:-1]} | {"amount": amount}


def _legit_transaction() -> dict[str, float]:
    # All-zero features, small amount: low fraud probability.
    return _transaction(v_scale=0.0, amount=20.0)


def _gray_zone_transaction() -> dict[str, float]:
    # Empirically sits in the fraud model's gray_zone band (~0.37 probability,
    # between the 0.30/0.70 policy thresholds).
    return _transaction(v_scale=1.36, amount=408.0)


def _high_risk_transaction() -> dict[str, float]:
    # Empirically well above the 0.70 high-risk threshold (~0.99 probability).
    return _transaction(v_scale=1.5, amount=450.0)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_welcome_message_with_docs_pointer():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


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
    # Low fraud probability + no sanctions hit is a clear-cut approve.
    assert body["confidence"] > 0.9
    assert "no_hit" in body["reasoning"]


def test_triage_escalates_on_confirmed_sanctions_match():
    case = {
        "entity_id": "cust_101",
        "full_name": "Anwar Al-Masri",
        "date_of_birth": "1975-03-14",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    body = response.json()
    assert body["decision"] == "escalate"
    # A confirmed match escalates even though the transaction itself is low-risk.
    assert body["confidence"] == 0.95
    assert "confirmed" in body["reasoning"].lower()


def test_triage_escalates_on_high_fraud_probability():
    case = {
        "entity_id": "cust_106",
        "full_name": "Taylor Reyes",
        "date_of_birth": "1995-04-20",
        "transaction": _high_risk_transaction(),
    }

    response = client.post("/triage", json=case)

    body = response.json()
    assert body["decision"] == "escalate"
    assert body["confidence"] > 0.9
    assert "no_hit" in body["reasoning"]


def test_triage_reviews_candidate_false_positive_sanctions_match():
    case = {
        "entity_id": "cust_102",
        "full_name": "Anwar Al-Masri",
        "date_of_birth": "1990-01-01",
        "transaction": _legit_transaction(),
    }

    response = client.post("/triage", json=case)

    body = response.json()
    # Name matches but DOB differs: a candidate false positive that must
    # never be auto-cleared, even though the transaction itself is low-risk.
    assert body["decision"] == "review"
    assert "candidate_false_positive" in body["reasoning"]


def test_triage_reviews_gray_zone_fraud_probability():
    case = {
        "entity_id": "cust_107",
        "full_name": "Morgan Lee",
        "date_of_birth": "1992-08-08",
        "transaction": _gray_zone_transaction(),
    }

    response = client.post("/triage", json=case)

    body = response.json()
    assert body["decision"] == "review"
    assert 0.0 < body["confidence"] < 1.0
    assert "no_hit" in body["reasoning"]


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
