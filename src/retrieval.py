"""Hybrid retrieval with explicit, inspectable scoring.

score = 0.40 * semantic + 0.25 * lexical + 0.15 * salience + 0.20 * decay

Only active memories compete. Superseded/expired never enter the candidate set.
"""

from __future__ import annotations

from datetime import datetime

from src.config import Settings
from src.decay import decay_multiplier
from src.embeddings import Embedder, cosine, lexical_overlap
from src.memory_store import MemoryStore
from src.models import Memory, MemoryStatus, ScoreBreakdown, utcnow


class MemoryRetriever:
    def __init__(self, store: MemoryStore, embedder: Embedder, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def score_all(self, query: str, *, now: datetime | None = None) -> list[ScoreBreakdown]:
        now = now or utcnow()
        query_vec = self.embedder.embed(query)
        scored: list[ScoreBreakdown] = []

        for memory in self.store.active():
            if memory.status != MemoryStatus.ACTIVE:
                continue
            semantic = cosine(query_vec, memory.embedding)
            lexical = max(
                lexical_overlap(query, memory.text()),
                lexical_overlap(query, memory.key),
                lexical_overlap(query, memory.value),
            )
            decay = decay_multiplier(memory, now)
            total = (
                self.settings.semantic_weight * semantic
                + self.settings.lexical_weight * lexical
                + self.settings.salience_weight * memory.salience
                + self.settings.decay_weight * decay
            )
            scored.append(
                ScoreBreakdown(
                    memory_id=memory.id,
                    key=memory.key,
                    value=memory.value,
                    semantic=round(semantic, 4),
                    lexical=round(lexical, 4),
                    salience=round(memory.salience, 4),
                    decay=round(decay, 4),
                    total=round(total, 4),
                    details={
                        "category": memory.category,
                        "status": memory.status,
                        "access_count": memory.access_count,
                    },
                )
            )

        scored.sort(key=lambda s: s.total, reverse=True)
        return scored

    def retrieve(
        self, query: str, *, now: datetime | None = None, limit: int | None = None
    ) -> list[ScoreBreakdown]:
        now = now or utcnow()
        limit = limit or self.settings.retrieval_limit
        scored = self.score_all(query, now=now)
        top = [s for s in scored if s.total > 0][:limit]
        self.store.touch([s.memory_id for s in top], when=now)
        return top

    def memories_for(self, scores: list[ScoreBreakdown]) -> list[Memory]:
        found: list[Memory] = []
        for score in scores:
            memory = self.store.get(score.memory_id)
            if memory:
                found.append(memory)
        return found
