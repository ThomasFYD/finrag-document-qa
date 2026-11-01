"""Layout extraction, source hashes, physical PDF page numbers, and provenance."""
import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_corpus(root: Path) -> tuple[list[Document], list[dict]]:
    entries = json.loads((root / "config/corpus.json").read_text())
    if not entries or len({e["doc_id"] for e in entries}) != len(entries):
        raise ValueError("corpus.json must be nonempty with unique doc_id values.")
    pages, sources = [], []
    for entry in entries:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", entry["doc_id"]):
            raise ValueError("doc_id may contain only letters, digits, underscores, and hyphens.")
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("PDF files must be inside the project directory.")
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("The demo limits each file to 50 MB.")
        digest = sha256_file(path)
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ValueError(f"{path.name} differs from its recorded source hash. Verify it before updating corpus.json.")
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported; supply a decrypted file.")
        if len(reader.pages) > 500:
            raise ValueError("The demo limits each file to 500 pages.")
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or ""
            if len(text.strip()) < 20:
                raise ValueError(f"{path.name}, page {i}, has insufficient extractable text and may need OCR. The page was not silently skipped.")
            pages.append(Document(page_content=text, metadata={
                "doc_id": entry["doc_id"], "source": entry["file"], "title": entry["title"],
                "period": entry.get("period", "unspecified"), "source_url": entry.get("source_url", ""),
                "source_sha256": digest, "page": i,
            }))
        sources.append({**entry, "sha256": digest, "pages": len(reader.pages)})
    return pages, sources
