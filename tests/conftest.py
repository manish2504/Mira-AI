from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import Settings
from src.conversation_store import ConversationStore
from src.db import connect
from src.embeddings import LocalHashEmbedder
from src.memory_store import MemoryStore
from src.models import Memory, MemoryStatus, utcnow
from src.retrieval import MemoryRetriever


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_path: Path):
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture
def memories(conn) -> MemoryStore:
    return MemoryStore(conn)


@pytest.fixture
def conversations(conn) -> ConversationStore:
    return ConversationStore(conn)


@pytest.fixture
def embedder() -> LocalHashEmbedder:
    return LocalHashEmbedder()


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash-lite",
        openai_embedding_model="text-embedding-3-small",
        gemini_base_url=None,
        embedding_provider="local",
        database_path=db_path,
        companion_provider="fake",
    )


@pytest.fixture
def retriever(memories, embedder, settings) -> MemoryRetriever:
    return MemoryRetriever(memories, embedder, settings)


def make_memory(
    embedder: LocalHashEmbedder,
    *,
    key: str,
    value: str,
    category: str = "personal",
    status: str = MemoryStatus.ACTIVE,
    salience: float = 0.7,
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    access_count: int = 0,
    supersedes: str | None = None,
) -> Memory:
    now = created_at or utcnow()
    text = f"{category} {key}: {value}"
    return Memory(
        id=str(uuid.uuid4()),
        category=category,
        key=key,
        value=value,
        source_conversation_id="c1",
        source_message_id="m1",
        confidence=0.9,
        salience=salience,
        created_at=now,
        last_accessed_at=last_accessed_at or now,
        access_count=access_count,
        status=status,
        supersedes_memory_id=supersedes,
        embedding=embedder.embed(text),
        embedding_model=embedder.model_name,
    )


def days_ago(n: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)
