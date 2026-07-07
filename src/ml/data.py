"""Synthetic stand-in for the Kaggle "Credit Card Fraud Detection" dataset.

That dataset (creditcard.csv) requires a Kaggle account/API token, which
isn't available in this environment. This generator mimics its shape instead:
anonymized PCA-style features V1..V28, a transaction amount, and a heavily
imbalanced binary label. Anything built against this generator (training,
tools) works unchanged if the real CSV is later dropped at
`data/creditcard.csv` — see `src.ml.train_fraud_model`.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = [f"v{i}" for i in range(1, 29)] + ["amount"]


def generate_synthetic_fraud_data(
    n_samples: int = 5000, fraud_rate: float = 0.02, random_state: int = 42
) -> pd.DataFrame:
    """Generate a synthetic, class-imbalanced transaction dataset.

    Fraudulent rows are drawn from a shifted, wider distribution than
    legitimate ones, so a classifier trained on this data has a genuine
    (but not trivial) signal to learn, similar to the real dataset.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    fraud_v = rng.normal(loc=1.5, scale=2.0, size=(n_fraud, 28))

    legit_amount = rng.lognormal(mean=3.0, sigma=1.0, size=n_legit)
    fraud_amount = rng.lognormal(mean=4.0, sigma=1.5, size=n_fraud)

    v_features = np.vstack([legit_v, fraud_v])
    amounts = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(v_features, columns=[f"v{i}" for i in range(1, 29)])
    df["amount"] = amounts
    df["class"] = labels.astype(int)

    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
