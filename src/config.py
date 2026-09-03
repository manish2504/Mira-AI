from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    gemini_base_url: str | None
    embedding_provider: str
    database_path: Path
    companion_provider: str
    openai_embedding_model: str = "text-embedding-3-small"
    recent_turn_limit: int = 16
    retrieval_limit: int = 12
    semantic_weight: float = 0.40
    lexical_weight: float = 0.25
    salience_weight: float = 0.15
    decay_weight: float = 0.20
    related_similarity_threshold: float = 0.55

    def __init__(
        self,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-3.5-flash-lite",
        gemini_base_url: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/",
        embedding_provider: str = "local",
        database_path: Path = Path("data/companion.db"),
        companion_provider: str = "gemini",
        openai_embedding_model: str = "text-embedding-3-small",
        recent_turn_limit: int = 16,
        retrieval_limit: int = 12,
        semantic_weight: float = 0.40,
        lexical_weight: float = 0.25,
        salience_weight: float = 0.15,
        decay_weight: float = 0.20,
        related_similarity_threshold: float = 0.55,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        openai_base_url: str | None = None,
    ):
        key = gemini_api_key or openai_api_key
        model = gemini_model if gemini_model != "gemini-3.5-flash-lite" else (openai_model or gemini_model)
        url = (
            gemini_base_url
            if gemini_base_url != "https://generativelanguage.googleapis.com/v1beta/openai/"
            else (openai_base_url or gemini_base_url)
        )
        object.__setattr__(self, "gemini_api_key", key)
        object.__setattr__(self, "gemini_model", model)
        object.__setattr__(self, "gemini_base_url", url)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "database_path", database_path)
        object.__setattr__(self, "companion_provider", companion_provider)
        object.__setattr__(self, "openai_embedding_model", openai_embedding_model)
        object.__setattr__(self, "recent_turn_limit", recent_turn_limit)
        object.__setattr__(self, "retrieval_limit", retrieval_limit)
        object.__setattr__(self, "semantic_weight", semantic_weight)
        object.__setattr__(self, "lexical_weight", lexical_weight)
        object.__setattr__(self, "salience_weight", salience_weight)
        object.__setattr__(self, "decay_weight", decay_weight)
        object.__setattr__(self, "related_similarity_threshold", related_similarity_threshold)

    @property
    def openai_api_key(self) -> str | None:
        return self.gemini_api_key

    @property
    def openai_model(self) -> str:
        return self.gemini_model

    @property
    def openai_base_url(self) -> str | None:
        return self.gemini_base_url


def load_settings() -> Settings:
    db = os.getenv("DATABASE_PATH", "data/companion.db")
    db_path = Path(db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or None
    base_url = (
        os.getenv("GEMINI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = (
        os.getenv("GEMINI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gemini-3.5-flash-lite"
    )
    provider = os.getenv("COMPANION_PROVIDER", "gemini").lower()

    return Settings(
        gemini_api_key=gemini_key,
        gemini_model=model,
        gemini_base_url=base_url,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
        database_path=db_path,
        companion_provider=provider,
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
    )
