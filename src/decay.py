"""Time decay: stale memories score lower; high salience / frequent access decay slower.

Superseded rows are excluded from retrieval entirely; decay does not resurrect them.
"""

from __future__ import annotations

from datetime import datetime

from src.models import Memory


def days_since(then: datetime, now: datetime) -> float:
    delta = now - then
    return max(0.0, delta.total_seconds() / 86400.0)


def half_life_days(memory: Memory) -> float:
    """Base ~7 days; high salience stretches toward ~28 days; recalls stretch further."""
    salience = max(0.0, min(1.0, memory.salience))
    access_boost = 1.0 + 0.12 * min(memory.access_count, 12)
    return (7.0 + 21.0 * salience) * access_boost


def decay_multiplier(memory: Memory, now: datetime) -> float:
    """Exponential half-life on last access (falls back to created_at)."""
    anchor = memory.last_accessed_at or memory.created_at
    age = days_since(anchor, now)
    hl = half_life_days(memory)
    return 0.5 ** (age / hl)
