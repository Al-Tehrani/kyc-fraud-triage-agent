from src.ml.data import FEATURE_NAMES
from src.tools.fraud_score import fraud_score


def _transaction(v_value: float, amount: float) -> dict[str, float]:
    return {name: v_value for name in FEATURE_NAMES[:-1]} | {"amount": amount}


def test_fraud_score_returns_valid_schema():
    result = fraud_score.invoke({"transaction": _transaction(v_value=0.0, amount=20.0)})

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_band"] in {"low", "gray_zone", "high"}
    assert len(result["top_contributing_features"]) == 3
    assert all("feature" in f and "contribution" in f for f in result["top_contributing_features"])


def test_fraud_score_ranks_fraud_like_pattern_above_legit_pattern():
    legit = fraud_score.invoke({"transaction": _transaction(v_value=0.0, amount=20.0)})
    fraud_like = fraud_score.invoke({"transaction": _transaction(v_value=3.0, amount=500.0)})

    assert fraud_like["fraud_probability"] > legit["fraud_probability"]
    assert legit["risk_band"] == "low"
