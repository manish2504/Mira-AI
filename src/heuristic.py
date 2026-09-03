"""Offline extraction used by FakeLLMProvider so the CLI demo works without an API.

Live chat uses the LLM extractor. These patterns cover the assessment walkthrough
and a few employment phrases — they are not a substitute for the model in production.
"""

from __future__ import annotations

import re

_FAVORITE = re.compile(
    r"my favorite (?P<key>[\w\s]+?) is (?P<value>[^.!?]+)",
    re.IGNORECASE,
)
_SWITCHED_DRINK = re.compile(
    r"switched to (?P<value>[\w\s]+?)(?: recently| lately)?[.!]?\s*$",
    re.IGNORECASE,
)
_WORK_AT = re.compile(
    r"(?:i work at|i'm at|i am at|joined)\s+(?P<company>[\w.& ]+?)(?:\s+and\b|[.!]|$)",
    re.IGNORECASE,
)
_LEFT_JOINED = re.compile(
    r"left (?P<old>[\w.& ]+?) and joined (?P<new>[\w.& ]+)",
    re.IGNORECASE,
)


def heuristic_extract(user_message: str) -> dict:
    memories: list[dict] = []
    text = user_message.strip()
    if not text:
        return {"memories": []}

    left = _LEFT_JOINED.search(text)
    if left:
        memories.append(
            _fact("work", "employer", left.group("new").strip(" ."), 0.95, 0.9)
        )
        return {"memories": memories}

    work = _WORK_AT.search(text)
    if work and not text.lower().startswith("where"):
        company = work.group("company").strip(" .")
        if company.lower() not in {"a", "the"}:
            memories.append(_fact("work", "employer", company, 0.9, 0.85))

    fav = _FAVORITE.search(text)
    if fav:
        key = "favorite " + " ".join(fav.group("key").strip().lower().split())
        value = fav.group("value").strip(" .")
        memories.append(_fact("preference", key, value, 0.95, 0.85))

    switched = _SWITCHED_DRINK.search(text)
    if switched and "favorite" not in text.lower():
        value = switched.group("value").strip(" .")
        memories.append(_fact("preference", "favorite drink", value, 0.9, 0.85))

    return {"memories": memories}


def _fact(category: str, key: str, value: str, confidence: float, salience: float) -> dict:
    return {
        "category": category,
        "key": key,
        "value": value,
        "confidence": confidence,
        "salience": salience,
    }
