from src.decay import decay_multiplier, half_life_days
from tests.conftest import days_ago, make_memory


def test_recent_memory_decays_less_than_stale(embedder):
    recent = make_memory(
        embedder,
        key="pet",
        value="cat",
        last_accessed_at=days_ago(1),
        salience=0.5,
    )
    stale = make_memory(
        embedder,
        key="pet",
        value="cat",
        last_accessed_at=days_ago(40),
        salience=0.5,
    )
    now = days_ago(0)
    assert decay_multiplier(recent, now) > decay_multiplier(stale, now)


def test_high_salience_decays_slower(embedder):
    important = make_memory(
        embedder,
        key="child name",
        value="Asha",
        last_accessed_at=days_ago(20),
        salience=1.0,
    )
    trivial = make_memory(
        embedder,
        key="snack",
        value="pretzels",
        last_accessed_at=days_ago(20),
        salience=0.1,
    )
    now = days_ago(0)
    assert half_life_days(important) > half_life_days(trivial)
    assert decay_multiplier(important, now) > decay_multiplier(trivial, now)


def test_frequent_access_slows_decay(embedder):
    cold = make_memory(
        embedder,
        key="city",
        value="Austin",
        last_accessed_at=days_ago(14),
        salience=0.5,
        access_count=0,
    )
    hot = make_memory(
        embedder,
        key="city",
        value="Austin",
        last_accessed_at=days_ago(14),
        salience=0.5,
        access_count=10,
    )
    now = days_ago(0)
    assert decay_multiplier(hot, now) > decay_multiplier(cold, now)


def test_decay_affects_retrieval_ranking(memories, embedder, retriever):
    stale_drink = make_memory(
        embedder,
        key="favorite drink",
        value="black coffee",
        category="preference",
        last_accessed_at=days_ago(60),
        salience=0.9,
    )
    # Would lose if we allowed superseded; both active on different keys.
    recent_city = make_memory(
        embedder,
        key="city",
        value="Austin",
        category="personal",
        last_accessed_at=days_ago(0.1),
        salience=0.4,
    )
    memories.insert(stale_drink)
    memories.insert(recent_city)
    ranked = retriever.score_all("tell me about me", now=days_ago(0))
    by_key = {s.key: s for s in ranked}
    assert by_key["city"].decay > by_key["favorite drink"].decay
