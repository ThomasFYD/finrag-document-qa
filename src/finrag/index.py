"""Versioned exact vector index, safe JSON/NumPy persistence, atomic promotion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from langchain_core.documents import Document
import numpy as np

from .chunking import VERSION, split_pages
from .ingest import read_corpus, sha256_file


def normalized(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid embedding: non-finite or zero-length vector")
    return array / norms


class VectorIndex:
    def __init__(self, documents, vectors, manifest):
        self.documents = documents
        self.vectors = normalized(vectors)
        self.manifest = manifest
        if self.vectors.ndim != 2 or len(documents) != len(self.vectors):
            raise ValueError("Text and vector counts do not match.")
        if self.vectors.shape[1] != manifest["dimension"]:
            raise ValueError("Index dimensions do not match.")

    def search_vector(self, vector, k: int, doc_id: str | None = None):
        q = normalized(vector)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError("Query and document embedding dimensions differ.")
        ids = [i for i, d in enumerate(self.documents) if not doc_id or d.metadata["doc_id"] == doc_id]
        scores = self.vectors @ q
        return [(i, float(scores[i])) for i in sorted(ids, key=lambda i: (-scores[i], i))[:k]]

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        target = directory / generation
        target.mkdir()
        docs = [{"text": d.page_content, "metadata": d.metadata} for d in self.documents]
        (target / "chunks.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
        np.save(target / "vectors.npy", self.vectors, allow_pickle=False)
        manifest = {**self.manifest, "generation": generation,
                    "chunks_sha256": sha256_file(target / "chunks.json"),
                    "vectors_sha256": sha256_file(target / "vectors.npy")}
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        pointer = directory / f".CURRENT-{generation}"
        pointer.write_text(generation)
        pointer.replace(directory / "CURRENT")
        self.manifest = manifest

    @classmethod
    def load(cls, directory: Path, embedding_fingerprint: str):
        try:
            generation = (directory / "CURRENT").read_text().strip()
        except FileNotFoundError as exc:
            raise ValueError("Index missing. Run finrag index first.") from exc
        if len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
            raise ValueError("Invalid index generation")
        target = directory / generation
        manifest = json.loads((target / "manifest.json").read_text())
        if manifest.get("schema") != 1 or manifest["embedding"] != embedding_fingerprint:
            raise ValueError("Index schema or embedding model changed. Rebuild the index.")
        for name, key in [("chunks.json", "chunks_sha256"), ("vectors.npy", "vectors_sha256")]:
            if sha256_file(target / name) != manifest[key]:
                raise ValueError(f"Index integrity check failed: {name}")
        documents = [Document(page_content=d["text"], metadata=d["metadata"]) for d in json.loads((target / "chunks.json").read_text())]
        return cls(documents, np.load(target / "vectors.npy", allow_pickle=False), manifest)


def build_indexes(settings, embeddings, methods=("baseline", "semantic")):
    pages, sources = read_corpus(settings.root)
    extracted = settings.root / "data/extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (extracted / f"{page.metadata['doc_id']}-p{page.metadata['page']:03}.txt").write_text(page.page_content)
    results = {}
    for method in methods:
        chunks = split_pages(pages, embeddings, settings, method)
        vectors = embeddings.embed_documents([d.page_content for d in chunks])
        manifest = {"schema": 1, "method": method, "chunker_version": VERSION,
                    "embedding": embeddings.fingerprint, "dimension": embeddings.dimension,
                    "chunk_tokens": settings.chunk_tokens, "overlap_tokens": settings.overlap_tokens,
                    "semantic_distance": settings.semantic_distance,
                    "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
                    "chunks": len(chunks), "pages": len(pages),
                    "corpus_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}
        index = VectorIndex(chunks, vectors, manifest)
        index.save(settings.root / "data/indexes" / method)
        results[method] = manifest
    return results
