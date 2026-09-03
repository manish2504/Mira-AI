"""Embeddings: OpenAI when configured, otherwise a stable local hashed vector."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

TOKEN_RE = re.compile(r"[a-z0-9]+")
LOCAL_DIM = 256
LOCAL_MODEL = "local-hash-v1"


class Embedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class LocalHashEmbedder:
    """Deterministic bag-of-tokens hashed into a fixed vector. No API, easy to explain."""

    model_name = LOCAL_MODEL

    def __init__(self, dim: int = LOCAL_DIM):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            h1 = int.from_bytes(digest[:4], "little")
            h2 = int.from_bytes(digest[4:8], "little")
            vec[h1 % self.dim] += 1.0
            vec[h2 % self.dim] += 0.5
        return l2_normalize(vec)


class OpenAIEmbedder:
    model_name: str

    def __init__(self, client, model: str):
        self.client = client
        self.model_name = model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return l2_normalize(list(response.data[0].embedding))


def lexical_overlap(query: str, document: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(document))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)
