from __future__ import annotations

import uuid
from datetime import datetime

from src.models import Message, utcnow


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class ConversationStore:
    def __init__(self, conn):
        self.conn = conn

    def ensure_conversation(self, conversation_id: str) -> None:
        row = self.conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
                (conversation_id, utcnow().isoformat()),
            )
            self.conn.commit()

    def add_message(
        self,
        conversation_id: str,
        session_id: str,
        role: str,
        content: str,
        created_at: datetime | None = None,
    ) -> Message:
        self.ensure_conversation(conversation_id)
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=created_at or utcnow(),
        )
        self.conn.execute(
            """
            INSERT INTO messages
                (id, conversation_id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                msg.id,
                msg.conversation_id,
                msg.session_id,
                msg.role,
                msg.content,
                msg.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return msg

    def recent(self, conversation_id: str, limit: int) -> list[Message]:
        rows = self.conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        messages = [
            Message(
                id=r["id"],
                conversation_id=r["conversation_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=_parse(r["created_at"]),
            )
            for r in reversed(rows)
        ]
        return messages
