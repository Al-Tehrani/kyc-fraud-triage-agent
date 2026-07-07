"""The `sanctions_check` tool: mock watchlist lookup against a synthetic list.

Per data/policies/sanctions_screening.md: a name+DOB match is a confirmed
hit; a name match with a differing DOB is a candidate false positive that
must be routed to a compliance analyst, never auto-cleared by the agent.
"""

import csv
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

WATCHLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "sanctions" / "watchlist.csv"


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


@lru_cache(maxsize=1)
def _load_watchlist() -> list[dict[str, str]]:
    with WATCHLIST_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


@tool
def sanctions_check(name: str, dob: str) -> dict:
    """Screen a name and date of birth (YYYY-MM-DD) against the sanctions/PEP watchlist.

    Returns `match_status` of "confirmed_match" (name and DOB both match),
    "candidate_false_positive" (name matches but DOB differs — requires
    analyst review, not an auto-clear), or "no_hit", plus the `list_type`
    ("SANCTIONS" or "PEP") and matched entry when applicable.
    """
    normalized_name = _normalize(name)
    for entry in _load_watchlist():
        if _normalize(entry["name"]) != normalized_name:
            continue
        match_status = "confirmed_match" if entry["dob"] == dob else "candidate_false_positive"
        return {"match_status": match_status, "list_type": entry["list_type"], "matched_entry": entry}
    return {"match_status": "no_hit", "list_type": None, "matched_entry": None}
