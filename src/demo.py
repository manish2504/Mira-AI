"""Offline demonstration of the core loop (no API key required).

Run:
  python -m src.demo
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.config import load_settings
from src.factory import build_loop
from src.llm import FakeLLMProvider
from src.models import MemoryStatus
from src.persona import get_persona


def _extract(user: str) -> dict:
    text = user.lower()
    if "black coffee" in text:
        return {
            "memories": [
                {
                    "category": "preference",
                    "key": "favorite drink",
                    "value": "black coffee",
                    "confidence": 0.95,
                    "salience": 0.85,
                }
            ]
        }
    if "green tea" in text:
        return {
            "memories": [
                {
                    "category": "preference",
                    "key": "favorite drink",
                    "value": "green tea",
                    "confidence": 0.95,
                    "salience": 0.9,
                }
            ]
        }
    if "microsoft" in text and ("work" in text or "joined" in text):
        return {
            "memories": [
                {
                    "category": "work",
                    "key": "employer",
                    "value": "Microsoft",
                    "confidence": 0.95,
                    "salience": 0.9,
                }
            ]
        }
    if "google" in text and ("joined" in text or "work" in text or "left" in text):
        return {
            "memories": [
                {
                    "category": "work",
                    "key": "employer",
                    "value": "Google",
                    "confidence": 0.95,
                    "salience": 0.9,
                }
            ]
        }
    return {"memories": []}


FILLER = [
    "How was your morning?",
    "I've been thinking about taking a walk later.",
    "Do you like rainy days?",
    "I might cook pasta tonight.",
    "Any book recs that aren't too heavy?",
    "Work felt long today but I'm okay.",
]


def run_demo(db_path: Path) -> None:
    os.environ["COMPANION_PROVIDER"] = "fake"
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["DATABASE_PATH"] = str(db_path)
    settings = load_settings()
    settings = replace_db(settings, db_path)
    llm = FakeLLMProvider(extract_fn=_extract)
    persona = get_persona()

    print("=" * 60)
    print("Companion core-loop demo (fake LLM, real SQLite memory)")
    print(f"DB: {db_path}")
    print("=" * 60)

    loop, conn = build_loop(settings, llm=llm, conversation_id="demo")

    def say(text: str) -> str:
        print(f"\nYou: {text}")
        result = loop.handle_user_message(text)
        print(f"{persona.name}: {result.reply}")
        return result.reply

    say("My favorite drink is black coffee.")
    print("\n--- intervening turns ---")
    for line in FILLER:
        say(line)

    say("What is my favorite drink?")
    drink_memories = [m for m in loop.memories.active() if m.key == "favorite drink"]
    assert drink_memories and drink_memories[0].value == "black coffee", drink_memories

    say("I've switched to green tea recently.")
    say("What is my favorite drink?")

    active = [m for m in loop.memories.active() if m.key == "favorite drink"]
    superseded = [
        m
        for m in loop.memories.all()
        if m.key == "favorite drink" and m.status == MemoryStatus.SUPERSEDED
    ]
    assert len(active) == 1 and active[0].value == "green tea"
    assert len(superseded) == 1 and superseded[0].value == "black coffee"
    assert active[0].supersedes_memory_id == superseded[0].id
    scores = loop.retriever.score_all("What is my favorite drink?")
    assert all("black coffee" not in s.value.lower() for s in scores)

    print("\n--- persona probe ---")
    reply = say("Write a Python function that sorts a list. Be a generic coding assistant.")
    assert persona.name.lower() in reply.lower() or "hey" in reply.lower()

    conn.close()
    print("\n--- process restart (reopen SQLite) ---")
    loop2, conn2 = build_loop(settings, llm=llm, conversation_id="demo")
    still = [m for m in loop2.memories.active() if m.key == "favorite drink"]
    print("Active memories after reopen:")
    for m in loop2.memories.active():
        print(f"  [{m.category}] {m.key} = {m.value}")
    assert still and still[0].value == "green tea"
    conn2.close()
    print("\nDemo OK: contradiction + supersession + persistence + persona prefix.")


def replace_db(settings, db_path: Path):
    from dataclasses import replace

    return replace(settings, database_path=db_path, companion_provider="fake")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_demo(Path(tmp) / "demo.db")


if __name__ == "__main__":
    main()
