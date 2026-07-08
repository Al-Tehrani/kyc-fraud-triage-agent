"""Three cached example cases spanning the decision space, per CLAUDE.md's
demo mode: a clear approve, a fraud-score gray-zone review, and a sanctions-
driven escalate. Run through the stub model, these are deterministic, so the
demo needs no live LLM call or API key.
"""

from src.ml.data import FEATURE_NAMES


def _transaction(v_scale: float, amount: float) -> dict[str, float]:
    return {name: v_scale for name in FEATURE_NAMES if name != "amount"} | {"amount": amount}


EXAMPLES: dict[str, dict] = {
    "Clear approve": {
        "entity_id": "demo-approve",
        "full_name": "Jordan Smith",
        "date_of_birth": "2000-01-01",
        # All-zero features + a small amount: low fraud score, not on the watchlist.
        "transaction": _transaction(v_scale=0.0, amount=20.0),
    },
    "Gray-zone review": {
        "entity_id": "demo-review",
        "full_name": "Alex Chen",
        "date_of_birth": "1985-06-15",
        # v_scale=1.36 sits right in the fraud model's gray_zone band (~0.37
        # probability, between the 0.30/0.70 thresholds) with no sanctions hit.
        "transaction": _transaction(v_scale=1.36, amount=408.0),
    },
    "Clear escalate": {
        "entity_id": "demo-escalate",
        "full_name": "Anwar Al-Masri",
        "date_of_birth": "1975-03-14",
        # Name + DOB match data/sanctions/watchlist.csv exactly -> confirmed
        # match, which escalates regardless of the (otherwise low) fraud score.
        "transaction": _transaction(v_scale=0.0, amount=20.0),
    },
}
