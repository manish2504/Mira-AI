from src.models import ExtractedFact, MemoryStatus
from src.update_engine import MemoryUpdateEngine


def engine(memories, embedder):
    return MemoryUpdateEngine(memories, embedder, llm=None)


def fact(key: str, value: str, category: str = "work") -> ExtractedFact:
    return ExtractedFact(
        category=category,
        key=key,
        value=value,
        confidence=0.9,
        salience=0.8,
    )


def test_insert_new_memory(memories, embedder):
    created = engine(memories, embedder).apply(
        [fact("employer", "Microsoft")],
        conversation_id="c1",
        message_id="m1",
    )
    assert len(created) == 1
    assert created[0].value == "Microsoft"
    assert created[0].status == MemoryStatus.ACTIVE


def test_duplicate_does_not_insert_second_row(memories, embedder):
    upd = engine(memories, embedder)
    upd.apply([fact("employer", "Microsoft")], conversation_id="c", message_id="1")
    upd.apply([fact("employer", "Microsoft")], conversation_id="c", message_id="2")
    rows = memories.all()
    assert len(rows) == 1
    assert rows[0].access_count == 1


def test_contradiction_supersedes_and_preserves_history(memories, embedder):
    upd = engine(memories, embedder)
    first = upd.apply(
        [fact("employer", "Microsoft")], conversation_id="c", message_id="1"
    )[0]
    second = upd.apply(
        [fact("employer", "Google")], conversation_id="c", message_id="2"
    )[0]

    old = memories.get(first.id)
    new = memories.get(second.id)
    assert old is not None and new is not None
    assert old.status == MemoryStatus.SUPERSEDED
    assert old.value == "Microsoft"
    assert new.status == MemoryStatus.ACTIVE
    assert new.value == "Google"
    assert new.supersedes_memory_id == old.id
    assert len(memories.active()) == 1


def test_preference_update_same_key(memories, embedder):
    upd = engine(memories, embedder)
    upd.apply(
        [fact("favorite drink", "black coffee", category="preference")],
        conversation_id="c",
        message_id="1",
    )
    upd.apply(
        [fact("favorite drink", "green tea", category="preference")],
        conversation_id="c",
        message_id="2",
    )
    active = memories.active()
    assert len(active) == 1
    assert active[0].value == "green tea"
    history = [m for m in memories.all() if m.status == MemoryStatus.SUPERSEDED]
    assert history[0].value == "black coffee"
