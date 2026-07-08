import json

from src.demo.app import _run_case
from src.demo.examples import EXAMPLES
from src.ml.data import FEATURE_NAMES


def test_examples_cover_all_required_features():
    for case in EXAMPLES.values():
        assert set(case["transaction"].keys()) == set(FEATURE_NAMES)


def test_clear_approve_example_approves():
    summary, raw = _run_case(json.dumps(EXAMPLES["Clear approve"]))

    assert json.loads(raw)["decision"] == "approve"
    assert "APPROVE" in summary


def test_gray_zone_example_reviews():
    summary, raw = _run_case(json.dumps(EXAMPLES["Gray-zone review"]))

    assert json.loads(raw)["decision"] == "review"
    assert "REVIEW" in summary


def test_clear_escalate_example_escalates():
    summary, raw = _run_case(json.dumps(EXAMPLES["Clear escalate"]))

    assert json.loads(raw)["decision"] == "escalate"
    assert "ESCALATE" in summary


def test_invalid_json_reports_error_without_raising():
    summary, raw = _run_case("not json")

    assert "Invalid JSON" in summary
    assert raw == ""


def test_missing_transaction_feature_reports_error_without_raising():
    case = json.loads(json.dumps(EXAMPLES["Clear approve"]))
    del case["transaction"]["v14"]

    summary, raw = _run_case(json.dumps(case))

    assert "Could not triage" in summary
    assert raw == ""
