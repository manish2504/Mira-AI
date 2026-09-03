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
- update: newer version of the EXACT SAME fact (e.g. same employer field, new company)
- contradict: cannot both be currently true about the EXACT SAME topic
- unrelated: different facts that can coexist; keep both active

CRITICAL RULES:
- "favorite drink: coffee" and "liked beverage: banana shake" are UNRELATED (different keys = different facts).
- "favorite drink: coffee" and "favorite drink: tea" are UPDATE (same key, new value).
- "employer: Google" and "employer: Microsoft" are UPDATE (same key, new value).
- "liked food: pizza" and "liked food: sushi" are UNRELATED (can like multiple things).
- Two facts are only update/contradict if they refer to the EXACT SAME attribute
  AND cannot logically coexist. Mere topical similarity (both about drinks, both
  about sports) does NOT make them updates of each other.

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
        # Only look for EXACT key matches in the same category first
        by_key = self.store.find_active_by_key(fact.category, fact.key)
        seen = {m.id for m in by_key}

        # Also check same key in other categories (e.g. "favorite drink" stored
        # under "preference" but new fact under "correction")
        extra = self.store.find_active_by_key_any_category(fact.key)
        for m in extra:
            if m.id not in seen:
                by_key.append(m)
                seen.add(m.id)

        # Only add semantic similarity matches if they have very high key similarity
        # (>= 0.85) to avoid false positives like "favorite drink" matching "liked beverage"
        fact_vec = self.embedder.embed(f"{fact.category} {fact.key}: {fact.value}")
        key_vec = self.embedder.embed(fact.key)
        for m in self.store.active():
            if m.id in seen:
                continue
            key_sim = cosine(key_vec, self.embedder.embed(m.key))
            value_sim = cosine(fact_vec, m.embedding)
            # Require VERY high key similarity (0.85+) to consider as candidate
            # This prevents "liked beverage" from matching "favorite drink"
            if key_sim >= 0.85 or (key_sim >= 0.72 and value_sim >= 0.70):
                by_key.append(m)
                seen.add(m.id)
        return by_key

    def _relate(self, existing: Memory, fact: ExtractedFact) -> RelationDecision:
        # Exact same key and value = duplicate
        if _norm(existing.value) == _norm(fact.value) and _norm(existing.key) == _norm(
            fact.key
        ):
            return RelationDecision.DUPLICATE

        # Same normalized key but different value = potential update
        # BUT only if it's a "singular" attribute (favorite, employer, etc.)
        # Not for "liked" items which can accumulate
        if _norm(existing.key) == _norm(fact.key) and _norm(existing.value) != _norm(
            fact.value
        ):
            # If the key starts with "liked" or "enjoys", these can coexist
            norm_key = _norm(fact.key)
            if norm_key.startswith("liked") or norm_key.startswith("enjoys"):
                return RelationDecision.UNRELATED
            # For singular attributes (favorite X, employer, name, etc.), it's an update
            return RelationDecision.UPDATE

        # Different keys = likely unrelated, but ask LLM if available
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
