"""Small deterministic text embedder that runs locally with no model download."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

DEFAULT_EMBEDDING_DIMENSIONS = 256
EMBEDDER_NAME = "feature-hash-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalEmbedder:
    """Map text to a normalized feature-hashed bag-of-words vector."""

    name = EMBEDDER_NAME

    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall(text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for normalized vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return float(sum(a * b for a, b in zip(left, right, strict=True)))
