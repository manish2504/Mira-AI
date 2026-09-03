from src.models import MemoryStatus
from tests.conftest import make_memory


def test_retrieval_ranks_relevant_memory_higher(memories, embedder, retriever):
    drink = make_memory(
        embedder,
        key="favorite drink",
        value="green tea",
        category="preference",
        salience=0.5,
    )
    job = make_memory(
        embedder, key="employer", value="Google", category="work", salience=0.9
    )
    memories.insert(drink)
    memories.insert(job)

    ranked = retriever.score_all("What is my favorite drink?")
    assert ranked[0].value == "green tea"
    assert ranked[0].total >= ranked[1].total


def test_superseded_memories_never_retrieved(memories, embedder, retriever):
    old = make_memory(
        embedder,
        key="favorite drink",
        value="black coffee",
        category="preference",
        status=MemoryStatus.SUPERSEDED,
        salience=1.0,
    )
    new = make_memory(
        embedder,
        key="favorite drink",
        value="green tea",
        category="preference",
        salience=0.4,
    )
    memories.insert(old)
    memories.insert(new)

    ranked = retriever.score_all("What is my favorite drink?")
    values = [s.value for s in ranked]
    assert "black coffee" not in values
    assert "green tea" in values


def test_retrieve_returns_only_top_k(memories, embedder, retriever, settings):
    for i in range(10):
        memories.insert(
            make_memory(embedder, key=f"note {i}", value=f"value {i}", salience=0.5)
        )
    top = retriever.retrieve("note", limit=3)
    assert len(top) == 3
