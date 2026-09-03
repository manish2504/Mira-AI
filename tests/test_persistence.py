from src.db import connect
from src.memory_store import MemoryStore
from tests.conftest import make_memory


def test_memories_survive_database_reopen(db_path, embedder):
    conn = connect(db_path)
    store = MemoryStore(conn)
    mem = make_memory(embedder, key="favorite drink", value="green tea", category="preference")
    store.insert(mem)
    conn.close()

    conn2 = connect(db_path)
    store2 = MemoryStore(conn2)
    loaded = store2.get(mem.id)
    conn2.close()

    assert loaded is not None
    assert loaded.value == "green tea"
    assert loaded.key == "favorite drink"


def test_messages_survive_database_reopen(db_path, conversations):
    conversations.add_message("default", "sess-1", "user", "hello")
    conversations.add_message("default", "sess-1", "assistant", "hey")
    # reopen via same file
    path = db_path
    conversations.conn.close()
    conn = connect(path)
    from src.conversation_store import ConversationStore

    store = ConversationStore(conn)
    recent = store.recent("default", 10)
    conn.close()
    assert [m.content for m in recent] == ["hello", "hey"]
    assert recent[0].session_id == "sess-1"
