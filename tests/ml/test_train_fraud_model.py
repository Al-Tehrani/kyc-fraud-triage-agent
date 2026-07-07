from src.ml.data import FEATURE_NAMES
from src.ml.train_fraud_model import train_model


def test_train_model_separates_classes_well(tmp_path, monkeypatch):
    import src.ml.train_fraud_model as train_module

    monkeypatch.setattr(train_module, "ARTIFACT_PATH", tmp_path / "fraud_model.json")

    model, metrics = train_model()

    assert model.n_features_in_ == len(FEATURE_NAMES)
    assert metrics["auc"] > 0.85
    assert (tmp_path / "fraud_model.json").exists()


def test_train_model_save_false_skips_artifact(tmp_path, monkeypatch):
    import src.ml.train_fraud_model as train_module

    monkeypatch.setattr(train_module, "ARTIFACT_PATH", tmp_path / "fraud_model.json")

    train_model(save=False)

    assert not (tmp_path / "fraud_model.json").exists()
