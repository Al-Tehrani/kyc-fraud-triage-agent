"""The `fraud_score` tool: XGBoost fraud probability with feature attribution."""

from functools import lru_cache

import pandas as pd
import xgboost as xgb
from langchain_core.tools import tool
from xgboost import XGBClassifier

from src.ml.data import FEATURE_NAMES
from src.ml.train_fraud_model import ARTIFACT_PATH, train_model

# Per data/policies/risk_score_thresholds.md
LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.70


@lru_cache(maxsize=1)
def _get_model() -> XGBClassifier:
    """Load the trained model from disk, or train one on the fly if missing."""
    if ARTIFACT_PATH.exists():
        model = XGBClassifier()
        model.load_model(str(ARTIFACT_PATH))
        return model
    model, _ = train_model()
    return model


def _risk_band(probability: float) -> str:
    if probability > HIGH_RISK_MIN:
        return "high"
    if probability >= LOW_RISK_MAX:
        return "gray_zone"
    return "low"


def _top_contributing_features(
    model: XGBClassifier, row: pd.DataFrame, top_n: int = 3
) -> list[dict[str, float | str]]:
    """Rank features by their SHAP contribution to this prediction."""
    dmatrix = xgb.DMatrix(row, feature_names=FEATURE_NAMES)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0][:-1]  # drop bias term
    ranked = sorted(zip(FEATURE_NAMES, contribs), key=lambda pair: abs(pair[1]), reverse=True)
    return [{"feature": name, "contribution": round(float(value), 4)} for name, value in ranked[:top_n]]


@tool
def fraud_score(transaction: dict[str, float]) -> dict:
    """Score a transaction's fraud probability with the trained XGBoost classifier.

    `transaction` must map each of v1..v28 (anonymized PCA-style features,
    mirroring the Kaggle creditcard.csv schema) and `amount` to a number.
    Returns the fraud probability, its policy risk band (low/gray_zone/high,
    per risk_score_thresholds.md), and the top contributing features by SHAP
    value for auditability.
    """
    model = _get_model()
    row = pd.DataFrame([[transaction[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    probability = float(model.predict_proba(row)[0, 1])
    return {
        "fraud_probability": round(probability, 4),
        "risk_band": _risk_band(probability),
        "top_contributing_features": _top_contributing_features(model, row),
    }
