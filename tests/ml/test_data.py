import pandas as pd

from src.ml.data import FEATURE_NAMES, generate_synthetic_fraud_data


def test_generate_synthetic_fraud_data_shape_and_imbalance():
    df = generate_synthetic_fraud_data(n_samples=1000, fraud_rate=0.05, random_state=0)

    assert len(df) == 1000
    assert set(FEATURE_NAMES + ["class"]) <= set(df.columns)
    assert 0.03 < df["class"].mean() < 0.07


def test_generate_synthetic_fraud_data_is_deterministic():
    df1 = generate_synthetic_fraud_data(random_state=1)
    df2 = generate_synthetic_fraud_data(random_state=1)

    pd.testing.assert_frame_equal(df1, df2)
