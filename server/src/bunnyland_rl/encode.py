"""Deterministic compact encoders for Bunnyland RL lenses."""

from __future__ import annotations

import hashlib
import re

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_RE.finditer(text))


def hashed_text_vector(
    text: str,
    *,
    dims: int = 64,
    ngrams: tuple[int, ...] = (1, 2),
) -> tuple[float, ...]:
    """Encode text into a stable signed hashing vector."""

    values = [0.0] * dims
    words = tokens(text)
    for ngram in ngrams:
        if ngram <= 0 or len(words) < ngram:
            continue
        for index in range(len(words) - ngram + 1):
            piece = " ".join(words[index : index + ngram])
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] & 1 else -1.0
            values[slot] += sign
    norm = max(1.0, sum(abs(value) for value in values) ** 0.5)
    return tuple(round(value / norm, 6) for value in values)


def stable_score(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


__all__ = ["hashed_text_vector", "stable_score", "tokens"]
