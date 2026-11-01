"""LangChain Embeddings adapter with E5 prefixes and a persistent exact cache."""
import hashlib
import json
import sqlite3
from threading import RLock

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings


class LocalEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        import torch
        from sentence_transformers import SentenceTransformer

        if not (settings.model_path / "config.json").exists():
            raise RuntimeError("Local model missing. Run python scripts/download_model.py.")
        record = json.loads((settings.model_path / "finrag-model.json").read_text())
        if record != {"model_id": settings.model_id, "revision": settings.model_revision}:
            raise ValueError("Local model revision differs from configuration. Download the configured model and rebuild indexes.")
        torch.set_num_threads(4)
        self.model = SentenceTransformer(str(settings.model_path), device="cpu", local_files_only=True,
                                         trust_remote_code=False)
        self.model.max_seq_length = 512
        self.tokenizer = self.model.tokenizer
        self.dimension = self.model.get_embedding_dimension()
        self.fingerprint = f"{settings.model_id}@{settings.model_revision}:e5-prefix:normalized:v1"
        self.cache = settings.cache_path
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        with sqlite3.connect(self.cache) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        inputs = [prefix + t for t in texts]
        if any(self.token_count(t) + 2 > 512 for t in inputs):
            raise ValueError("Text exceeds the E5 512-token limit. Silent truncation is disabled; shorten the query or adjust chunking.")
        keys = [hashlib.sha256((self.fingerprint + "\0" + t).encode()).hexdigest() for t in inputs]
        with self.lock, sqlite3.connect(self.cache, timeout=30) as conn:
            vectors = {}
            missing = {}
            for key, text in zip(keys, inputs):
                row = conn.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    vectors[key] = np.frombuffer(row[0], dtype=np.float32).copy()
                else:
                    missing[key] = text
            if missing:
                encoded = self.model.encode(list(missing.values()), batch_size=16, normalize_embeddings=True,
                                            show_progress_bar=False, convert_to_numpy=True)
                for key, vector in zip(missing, encoded):
                    vector = np.asarray(vector, dtype=np.float32)
                    vectors[key] = vector
                    conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, vector.tobytes()))
            return [vectors[key].tolist() for key in keys]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query: ")[0]
