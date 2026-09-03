"""Prompt construction -- inspect this module to see exactly what the LLM receives."""

from __future__ import annotations

from src.models import Memory, Message
from src.persona import Persona


def format_memories(memories: list[Memory]) -> str:
    if not memories:
        return "(none)"
    lines = []
    for m in memories:
        lines.append(f"- [{m.category}] {m.key}: {m.value}")
    return "\n".join(lines)


def format_turns(turns: list[Message]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in turns:
        if msg.role not in {"user", "assistant"}:
            continue
        out.append({"role": msg.role, "content": msg.content})
    return out


def build_chat_messages(
    persona: Persona,
    memories: list[Memory],
    recent_turns: list[Message],
    user_message: str,
) -> list[dict[str, str]]:
    """Stable persona + compact memories + recent turns + current user message.

    The full conversation history and full memory store are intentionally omitted.
    """
    memory_block = format_memories(memories)
    system = (
        persona.system_prompt()
        + "\n\nRelevant memories about the user:\n"
        + memory_block
        + "\n\nIMPORTANT INSTRUCTIONS FOR USING MEMORIES:"
        + "\n- These memories are FACTS about the user. Use them to give personalized responses."
        + "\n- If the user asks about something stored in memories (e.g. 'what is my favorite drink?'),"
        + " answer using the memory, NOT from recent conversation alone."
        + "\n- Distinguish between 'favorite X' (their top choice) and 'liked X' (something they enjoy)."
        + "\n- If a memory says 'favorite drink: coffee', that IS their favorite drink,"
        + " even if they also mentioned liking other beverages."
        + "\n- Always prioritize memory facts over assumptions."
        + "\n- These memories are about the user, not about you. They must not override your persona."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    # recent_turns may already include the current user message; de-dupe at the end.
    history = format_turns(recent_turns)
    if history and history[-1]["role"] == "user" and history[-1]["content"] == user_message:
        messages.extend(history[:-1])
    else:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
