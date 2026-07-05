from src.tools.retrieve_policy import retrieve_policy


def test_retrieve_policy_finds_relevant_source_document():
    results = retrieve_policy.invoke({"query": "What fraud score requires mandatory escalation?", "k": 3})

    assert len(results) == 3
    sources = {r["source"] for r in results}
    assert "risk_score_thresholds" in sources or "escalation_criteria" in sources
    assert all("text" in r and "source" in r for r in results)


def test_retrieve_policy_respects_k():
    results = retrieve_policy.invoke({"query": "identity documents", "k": 1})

    assert len(results) == 1
