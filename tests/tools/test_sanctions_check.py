from src.tools.sanctions_check import sanctions_check


def test_confirmed_match_on_sanctions_list():
    result = sanctions_check.invoke({"name": "Anwar Al-Masri", "dob": "1975-03-14"})

    assert result["match_status"] == "confirmed_match"
    assert result["list_type"] == "SANCTIONS"


def test_confirmed_match_on_pep_list():
    result = sanctions_check.invoke({"name": "Dmitri Volkov", "dob": "1968-07-22"})

    assert result["match_status"] == "confirmed_match"
    assert result["list_type"] == "PEP"


def test_name_match_with_differing_dob_is_candidate_false_positive():
    result = sanctions_check.invoke({"name": "Anwar Al-Masri", "dob": "1990-01-01"})

    assert result["match_status"] == "candidate_false_positive"
    assert result["list_type"] == "SANCTIONS"


def test_no_hit_for_unlisted_name():
    result = sanctions_check.invoke({"name": "Jordan Smith", "dob": "2000-01-01"})

    assert result["match_status"] == "no_hit"
    assert result["list_type"] is None
    assert result["matched_entry"] is None


def test_match_is_case_and_whitespace_insensitive():
    result = sanctions_check.invoke({"name": "  ANWAR   al-masri  ", "dob": "1975-03-14"})

    assert result["match_status"] == "confirmed_match"
