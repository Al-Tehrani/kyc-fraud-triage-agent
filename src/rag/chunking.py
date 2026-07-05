"""Turn markdown policy docs into retrievable chunks, one per `## ` section."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # policy doc filename, without extension


def chunk_document(text: str, source: str) -> list[Chunk]:
    """Split a markdown doc along its '## ' section headers.

    Each section (heading + body) becomes one chunk, so a retrieved chunk
    is always a self-contained, citable unit of policy.
    """
    sections = re.split(r"\n(?=## )", text.strip())
    return [Chunk(text=section.strip(), source=source) for section in sections if section.startswith("## ")]


def chunk_policy_docs(policies_dir: Path) -> list[Chunk]:
    """Load every *.md file in `policies_dir` and chunk it."""
    chunks = []
    for path in sorted(policies_dir.glob("*.md")):
        chunks.extend(chunk_document(path.read_text(), source=path.stem))
    return chunks
