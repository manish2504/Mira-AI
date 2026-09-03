from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class MemoryCategory(StrEnum):
    PREFERENCE = "preference"
    WORK = "work"
    PLAN = "plan"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    PERSONAL = "personal"
    CORRECTION = "correction"


class ExtractedFact(BaseModel):
    category: str
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    salience: float = Field(ge=0.0, le=1.0, default=0.6)

    @field_validator("category", "key", "value")
    @classmethod
    def strip_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("key")
    @classmethod
    def normalize_key(cls, v: str) -> str:
        return " ".join(v.lower().replace("-", " ").split())


class ExtractionResult(BaseModel):
    memories: list[ExtractedFact] = Field(default_factory=list)


class RelationDecision(StrEnum):
    DUPLICATE = "duplicate"
    UPDATE = "update"
    CONTRADICT = "contradict"
    UNRELATED = "unrelated"


class RelationResult(BaseModel):
    decision: RelationDecision
    reason: str = ""


@dataclass
class Message:
    id: str
    conversation_id: str
    session_id: str
    role: str
    content: str
    created_at: datetime


@dataclass
class Memory:
    id: str
    category: str
    key: str
    value: str
    source_conversation_id: str | None
    source_message_id: str | None
    confidence: float
    salience: float
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    status: str
    supersedes_memory_id: str | None
    embedding: list[float] | None = None
    embedding_model: str | None = None

    def text(self) -> str:
        return f"{self.category} {self.key}: {self.value}"


@dataclass
class ScoreBreakdown:
    memory_id: str
    key: str
    value: str
    semantic: float
    lexical: float
    salience: float
    decay: float
    total: float
    details: dict[str, Any] = field(default_factory=dict)
