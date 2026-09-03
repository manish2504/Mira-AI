"""Contradiction / supersession: never delete; mark old rows superseded."""

from __future__ import annotations

import json
import uuid

from src.embeddings import Embedder, cosine
from src.llm import LLMProvider
from src.memory_store import MemoryStore
from src.models import (
    ExtractedFact,
    Memory,
    MemoryStatus,
    RelationDecision,
    RelationResult,
    utcnow,
)

CLASSIFY_SYSTEM = """Classify the relationship between an EXISTING memory and a NEW candidate fact.

Return JSON: {"decision": "duplicate|update|contradict|unrelated", "reason": "short"}

- duplicate: same information, wording may differ
- update: newer version of the same fact (e.g. still the employer, new company)
- contradict: cannot both be currently true
- unrelated: different facts; keep both active

update and contradict both mean the old memory should be superseded.
"""


class MemoryUpdateEngine:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        llm: LLMProvider | None = None,
        related_threshold: float = 0.55,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.related_threshold = related_threshold

    def apply(
        self,
        facts: list[ExtractedFact],
        *,
        conversation_id: str | None,
        message_id: str | None,
    ) -> list[Memory]:
        created: list[Memory] = []
        for fact in facts:
            created.extend(
                self._apply_one(fact, conversation_id=conversation_id, message_id=message_id)
            )
        return created

    def _apply_one(
        self,
        fact: ExtractedFact,
        *,
        conversation_id: str | None,
        message_id: str | None,
    ) -> list[Memory]:
        candidates = self._candidates(fact)
        now = utcnow()
        embedding = self.embedder.embed(f"{fact.category} {fact.key}: {fact.value}")

        superseded_ids: list[str] = []
        for existing in candidates:
            decision = self._relate(existing, fact)
            if decision == RelationDecision.UNRELATED:
                continue
            if decision == RelationDecision.DUPLICATE:
                # Refresh usefulness; do not insert a twin.
                self.store.touch([existing.id], when=now)
                return []
            # update or contradict
            self.store.mark_superseded(existing.id)
            superseded_ids.append(existing.id)

        new_memories: list[Memory] = []
        if not superseded_ids:
            new_memories.append(
                self._insert(
                    fact,
                    embedding=embedding,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    supersedes=None,
                    now=now,
                )
            )
            return new_memories

        # One new active row that points at the most recent superseded memory.
        parent = superseded_ids[-1]
        new_memories.append(
            self._insert(
                fact,
                embedding=embedding,
                conversation_id=conversation_id,
                message_id=message_id,
                supersedes=parent,
                now=now,
            )
        )
        return new_memories

    def _insert(
        self,
        fact: ExtractedFact,
        *,
        embedding: list[float],
        conversation_id: str | None,
        message_id: str | None,
        supersedes: str | None,
        now,
    ) -> Memory:
        memory = Memory(
            id=str(uuid.uuid4()),
            category=fact.category,
            key=fact.key,
            value=fact.value,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            confidence=fact.confidence,
            salience=fact.salience,
            created_at=now,
            last_accessed_at=now,
            access_count=0,
            status=MemoryStatus.ACTIVE,
            supersedes_memory_id=supersedes,
            embedding=embedding,
            embedding_model=self.embedder.model_name,
        )
        return self.store.insert(memory)

    def _candidates(self, fact: ExtractedFact) -> list[Memory]:
        by_key = self.store.find_active_by_key(fact.category, fact.key)
        seen = {m.id for m in by_key}
        extra = self.store.find_active_by_key_any_category(fact.key)
        for m in extra:
            if m.id not in seen:
                by_key.append(m)
                seen.add(m.id)

        fact_vec = self.embedder.embed(f"{fact.category} {fact.key}: {fact.value}")
        key_vec = self.embedder.embed(fact.key)
        for m in self.store.active():
            if m.id in seen:
                continue
            same_keyish = cosine(key_vec, self.embedder.embed(m.key)) >= 0.72
            value_sim = cosine(fact_vec, m.embedding)
            if same_keyish or value_sim >= self.related_threshold:
                by_key.append(m)
                seen.add(m.id)
        return by_key

    def _relate(self, existing: Memory, fact: ExtractedFact) -> RelationDecision:
        if _norm(existing.value) == _norm(fact.value) and _norm(existing.key) == _norm(
            fact.key
        ):
            return RelationDecision.DUPLICATE
        if _norm(existing.key) == _norm(fact.key) and _norm(existing.value) != _norm(
            fact.value
        ):
            return RelationDecision.UPDATE

        if self.llm is None:
            return RelationDecision.UNRELATED

        payload = {
            "existing": {
                "category": existing.category,
                "key": existing.key,
                "value": existing.value,
            },
            "new": {
                "category": fact.category,
                "key": fact.key,
                "value": fact.value,
            },
        }
        raw = self.llm.complete(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            json_mode=True,
            temperature=0.0,
        )
        try:
            result = RelationResult.model_validate(json.loads(raw))
            return result.decision
        except (ValueError, Exception):
            return RelationDecision.UNRELATED


def _norm(text: str) -> str:
    return " ".join(text.lower().split())
