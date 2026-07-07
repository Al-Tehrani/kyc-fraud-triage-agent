"""Train the XGBoost fraud classifier.

Trains against the real Kaggle "Credit Card Fraud Detection" dataset if
present at `data/creditcard.csv`. That dataset isn't available in this
environment (it requires a Kaggle account), so training falls back to the
synthetic generator in `src.ml.data`, which mirrors its schema (v1..v28 +
amount + class). Dropping the real CSV in at that path retrains on real data
with no code changes.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.ml.data import FEATURE_NAMES, generate_synthetic_fraud_data

REAL_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "fraud_model.json"


def _load_dataset() -> pd.DataFrame:
    if REAL_DATA_PATH.exists():
        df = pd.read_csv(REAL_DATA_PATH)
        df.columns = [c.lower() for c in df.columns]
        return df[FEATURE_NAMES + ["class"]]
    return generate_synthetic_fraud_data()


def train_model(save: bool = True) -> tuple[XGBClassifier, dict[str, float]]:
    """Train the classifier and return it along with holdout metrics."""
    df = _load_dataset()
    X, y = df[FEATURE_NAMES], df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }

    if save:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(ARTIFACT_PATH))

    return model, metrics


if __name__ == "__main__":
    _, metrics = train_model()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
