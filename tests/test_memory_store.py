from src.models import MemoryStatus
from tests.conftest import make_memory


def test_insert_and_get(memories, embedder):
    mem = make_memory(embedder, key="employer", value="Microsoft", category="work")
    memories.insert(mem)
    loaded = memories.get(mem.id)
    assert loaded is not None
    assert loaded.key == "employer"
    assert loaded.value == "Microsoft"
    assert loaded.status == MemoryStatus.ACTIVE
    assert loaded.embedding is not None


def test_active_excludes_superseded(memories, embedder):
    a = make_memory(embedder, key="employer", value="Microsoft", category="work")
    b = make_memory(
        embedder,
        key="employer",
        value="Google",
        category="work",
        status=MemoryStatus.SUPERSEDED,
    )
    memories.insert(a)
    memories.insert(b)
    active = memories.active()
    assert [m.value for m in active] == ["Microsoft"]
