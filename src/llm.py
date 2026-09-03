from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI

from src.config import Settings


class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.4,
    ) -> str: ...


class GeminiProvider:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set GEMINI_API_KEY in .env or environment, "
                "or use COMPANION_PROVIDER=fake for the offline demo."
            )
        kwargs: dict = {"api_key": settings.gemini_api_key}
        if settings.gemini_base_url:
            kwargs["base_url"] = settings.gemini_base_url
        self.client = OpenAI(**kwargs)
        self.model = settings.gemini_model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.4,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "insufficient_quota" in str(exc) or "credit_balance_exhausted" in str(exc):
                raise RuntimeError(
                    "API quota exhausted or invalid billing status. "
                    "Check your API key or run in offline mode with: $env:COMPANION_PROVIDER='fake'"
                ) from exc
            raise RuntimeError(f"Gemini API error: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty completion")
        return content


# Backwards compatibility alias
OpenAIProvider = GeminiProvider


class FakeLLMProvider:
    """Deterministic provider for tests and the offline demo.

    Extraction/classification can be scripted via `extract_fn` / `classify_fn`.
    Chat replies mention retrieved memories so recall is observable without an API.
    """

    def __init__(self, extract_fn=None, classify_fn=None, chat_fn=None):
        self.extract_fn = extract_fn
        self.classify_fn = classify_fn
        self.chat_fn = chat_fn

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.4,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if json_mode and "Extract durable memories" in system:
            fn = self.extract_fn
            if fn is None:
                from src.heuristic import heuristic_extract

                fn = heuristic_extract
            return json.dumps(fn(user))

        if json_mode and "Classify the relationship" in system:
            if self.classify_fn:
                return json.dumps(self.classify_fn(user))
            return json.dumps({"decision": "unrelated", "reason": "default fake"})

        if self.chat_fn:
            return self.chat_fn(messages)
        return _default_fake_chat(messages)


def _default_fake_chat(messages: list[dict[str, str]]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    memories_block = ""
    if "Relevant memories about the user:" in system:
        memories_block = system.split("Relevant memories about the user:", 1)[1]
        memories_block = memories_block.split("These memories are about the user", 1)[0]
    name = "Mira"
    if "You are " in system:
        name = system.split("You are ", 1)[1].split(",", 1)[0].strip()
    snippet = memories_block.strip()
    if snippet and snippet != "(none)":
        return (
            f"Hey - {name} here. I remember: {snippet.splitlines()[0].strip()} "
            f"Anyway, about what you just said: {user[:120]}"
        )
    return (
        f"Hey, it's {name}. I'm with you on that. {user[:80]} "
        "Want to tell me a bit more?"
    )


def build_provider(settings: Settings) -> LLMProvider:
    if settings.companion_provider == "fake":
        return FakeLLMProvider()
    return GeminiProvider(settings)
