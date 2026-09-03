from __future__ import annotations

import sys

from src.config import load_settings
from src.factory import build_loop
from src.persona import get_persona


HELP = """Commands:
  /memories   show active memories
  /scores     show last retrieval score breakdown
  /help       this text
  /quit       exit
"""


def main() -> None:
    settings = load_settings()
    persona = get_persona()
    loop, conn = build_loop(settings)
    print(f"{persona.name} is here. Memories persist in {settings.database_path}")
    print(f"Provider: {settings.companion_provider}  |  embeddings: {settings.embedding_provider}")
    print("Type /help for commands.\n")
    try:
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not text:
                continue
            if text.lower() in {"/quit", "/exit"}:
                print("Bye.")
                break
            if text.lower() in {"/help"}:
                print(HELP)
                continue
            if text.lower() in {"/memories"}:
                _print_memories(loop)
                continue
            if text.lower() in {"/scores"}:
                _print_scores(loop)
                continue
            result = loop.handle_user_message(text)
            print(f"{persona.name}: {result.reply}\n")
    finally:
        conn.close()


def _print_memories(loop) -> None:
    rows = loop.memories.active()
    if not rows:
        print("(no active memories)\n")
        return
    print("Active memories:")
    for m in rows:
        print(f"  [{m.category}] {m.key} = {m.value}  (salience={m.salience:.2f})")
    print()


def _print_scores(loop) -> None:
    if not loop.last_retrieval:
        print("(no retrieval yet — send a message first)\n")
        return
    print("Last retrieval (highest first):")
    print("  total = 0.40*semantic + 0.25*lexical + 0.15*salience + 0.20*decay")
    for s in loop.last_retrieval:
        print(
            f"  {s.total:.3f}  {s.key}={s.value}  "
            f"(sem={s.semantic:.2f} lex={s.lexical:.2f} "
            f"sal={s.salience:.2f} decay={s.decay:.2f})"
        )
    print()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
