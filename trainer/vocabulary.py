"""
Vocabulary data structure for TokeNismo Unigram LM trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class VocabEntry:
    token: bytes
    log_prob: float
    token_id: int


class Vocabulary:
    """
    Ordered vocabulary mapping bytes → (token_id, log_prob).
    Token IDs are assigned in insertion order.
    """

    def __init__(self) -> None:
        self._token_to_id: dict[bytes, int] = {}
        self._entries: list[VocabEntry] = []

    @classmethod
    def from_log_probs(cls, log_probs: dict[bytes, float]) -> "Vocabulary":
        vocab = cls()
        for token, lp in sorted(log_probs.items(), key=lambda x: -x[1]):
            vocab._add(token, lp)
        return vocab

    def _add(self, token: bytes, log_prob: float) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        token_id = len(self._entries)
        self._token_to_id[token] = token_id
        self._entries.append(VocabEntry(token, log_prob, token_id))
        return token_id

    def get_id(self, token: bytes) -> int | None:
        return self._token_to_id.get(token)

    def get_entry(self, token_id: int) -> VocabEntry:
        return self._entries[token_id]

    def log_prob(self, token: bytes) -> float:
        tid = self._token_to_id.get(token)
        if tid is None:
            return float("-inf")
        return self._entries[tid].log_prob

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, token: bytes) -> bool:
        return token in self._token_to_id

    def __iter__(self) -> Iterator[VocabEntry]:
        return iter(self._entries)

    def items(self) -> Iterator[tuple[bytes, float]]:
        for entry in self._entries:
            yield entry.token, entry.log_prob
