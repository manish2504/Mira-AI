from __future__ import annotations

import json
import uuid
from datetime import datetime

from src.models import Memory, MemoryStatus, utcnow


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _embedding_from_row(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    data = json.loads(raw)
    return [float(x) for x in data]


class MemoryStore:
    def __init__(self, conn):
        self.conn = conn

    def insert(self, memory: Memory) -> Memory:
        if not memory.id:
            memory.id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO memories (
                id, category, key, value, source_conversation_id, source_message_id,
                confidence, salience, created_at, last_accessed_at, access_count,
                status, supersedes_memory_id, embedding, embedding_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.category,
                memory.key,
                memory.value,
                memory.source_conversation_id,
                memory.source_message_id,
                memory.confidence,
                memory.salience,
                memory.created_at.isoformat(),
                memory.last_accessed_at.isoformat(),
                memory.access_count,
                memory.status,
                memory.supersedes_memory_id,
                json.dumps(memory.embedding) if memory.embedding is not None else None,
                memory.embedding_model,
            ),
        )
        self.conn.commit()
        return memory

    def get(self, memory_id: str) -> Memory | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def list_by_status(self, status: str) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE status = ? ORDER BY created_at ASC",
            (status,),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def active(self) -> list[Memory]:
        return self.list_by_status(MemoryStatus.ACTIVE)

    def find_active_by_key(self, category: str, key: str) -> list[Memory]:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE status = ? AND lower(category) = lower(?) AND key = ?
            """,
            (MemoryStatus.ACTIVE, category, key),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def find_active_by_key_any_category(self, key: str) -> list[Memory]:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE status = ? AND key = ?
            """,
            (MemoryStatus.ACTIVE, key),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def all(self) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY created_at ASC"
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def mark_superseded(self, memory_id: str) -> None:
        self.conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?",
            (MemoryStatus.SUPERSEDED, memory_id),
        )
        self.conn.commit()

    def touch(self, memory_ids: list[str], when: datetime | None = None) -> None:
        if not memory_ids:
            return
        ts = (when or utcnow()).isoformat()
        for mid in memory_ids:
            self.conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (ts, mid),
            )
        self.conn.commit()

    def _from_row(self, row) -> Memory:
        return Memory(
            id=row["id"],
            category=row["category"],
            key=row["key"],
            value=row["value"],
            source_conversation_id=row["source_conversation_id"],
            source_message_id=row["source_message_id"],
            confidence=row["confidence"],
            salience=row["salience"],
            created_at=_parse(row["created_at"]),
            last_accessed_at=_parse(row["last_accessed_at"]),
            access_count=row["access_count"],
            status=row["status"],
            supersedes_memory_id=row["supersedes_memory_id"],
            embedding=_embedding_from_row(row["embedding"]),
            embedding_model=row["embedding_model"],
        )
