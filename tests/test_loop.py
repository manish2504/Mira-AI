from src.factory import build_loop
from src.llm import FakeLLMProvider
from src.models import MemoryStatus
from src.demo import _extract


def test_core_loop_drink_update_and_recall(settings):
    llm = FakeLLMProvider(extract_fn=_extract)
    loop, conn = build_loop(settings, llm=llm, conversation_id="t")
    try:
        loop.handle_user_message("My favorite drink is black coffee.")
        loop.handle_user_message("How was your morning?")
        loop.handle_user_message("What is my favorite drink?")
        first = [m.value for m in loop.memories.active() if m.key == "favorite drink"]
        assert first == ["black coffee"]

        loop.handle_user_message("I've switched to green tea recently.")
        result = loop.handle_user_message("What is my favorite drink?")
        active = [m for m in loop.memories.active() if m.key == "favorite drink"]
        assert len(active) == 1
        assert active[0].value == "green tea"
        assert "green tea" in result.reply.lower() or "favorite drink" in result.reply.lower()

        old = [m for m in loop.memories.all() if m.status == MemoryStatus.SUPERSEDED]
        assert old[0].value == "black coffee"
        ranked = loop.retriever.score_all("What is my favorite drink?")
        assert all(s.value != "black coffee" for s in ranked)
    finally:
        conn.close()
