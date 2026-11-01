"""Configuration is explicit; importing this module never calls an API."""
from dataclasses import dataclass
from pathlib import Path
import json
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    model_id: str = "intfloat/multilingual-e5-small"
    model_revision: str = ""
    chunk_tokens: int = 360
    overlap_tokens: int = 48
    semantic_distance: float = 0.18
    context_bytes: int = 14000
    top_k: int = 5

    @classmethod
    def load(cls, root: Path = ROOT):
        load_dotenv(root / ".env", override=False)
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        config = json.loads((root / "config/settings.json").read_text())
        return cls(root=root, **config)

    @property
    def model_path(self):
        return self.root / "data/models/multilingual-e5-small"

    @property
    def cache_path(self):
        return self.root / "data/cache/embeddings.sqlite3"

    @property
    def remote_enabled(self):
        return bool(os.getenv("DEEPINFRA_API_KEY", "").strip())
