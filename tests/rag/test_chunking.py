from pathlib import Path

from src.rag.chunking import chunk_document, chunk_policy_docs

POLICIES_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"


def test_chunk_document_splits_on_section_headers():
    text = "# Title\n\nIntro line.\n\n## Section One\n\nBody one.\n\n## Section Two\n\nBody two.\n"

    chunks = chunk_document(text, source="doc")

    assert len(chunks) == 2
    assert chunks[0].text.startswith("## Section One")
    assert chunks[1].text.startswith("## Section Two")
    assert all(chunk.source == "doc" for chunk in chunks)


def test_chunk_policy_docs_loads_all_five_real_files():
    chunks = chunk_policy_docs(POLICIES_DIR)

    sources = {chunk.source for chunk in chunks}
    assert sources == {
        "identity_verification",
        "risk_score_thresholds",
        "sanctions_screening",
        "enhanced_due_diligence",
        "escalation_criteria",
    }
    assert len(chunks) > len(sources)  # each doc has multiple sections
