"""
Trie for fast prefix lookups during Viterbi encoding.
Used by both the Python prototype and as a reference for the Rust port.
"""

from __future__ import annotations
from typing import Optional


class TrieNode:
    __slots__ = ("children", "token_id", "log_prob")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.token_id: Optional[int] = None
        self.log_prob: float = float("-inf")


class Trie:
    """
    Byte-level trie. Keys are bytes objects; values are (token_id, log_prob).
    Supports prefix iteration: walk byte-by-byte, collecting all terminal nodes.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    def insert(self, token: bytes, log_prob: float) -> int:
        token_id = self._size
        node = self.root
        for byte in token:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        node.token_id = token_id
        node.log_prob = log_prob
        self._size += 1
        return token_id

    def get(self, token: bytes) -> Optional[int]:
        node = self.root
        for byte in token:
            node = node.children.get(byte)
            if node is None:
                return None
        return node.token_id

    def walk(self, data: bytes, start: int) -> list[tuple[int, int, float]]:
        """
        Walk trie from position `start` in `data`.
        Returns list of (end_pos, token_id, log_prob) for each token found.
        end_pos is exclusive (i.e. data[start:end_pos] is the token).
        """
        results: list[tuple[int, int, float]] = []
        node = self.root
        for i in range(start, len(data)):
            byte = data[i]
            node = node.children.get(byte)
            if node is None:
                break
            if node.token_id is not None:
                results.append((i + 1, node.token_id, node.log_prob))
        return results

    def __len__(self) -> int:
        return self._size

    def __contains__(self, token: bytes) -> bool:
        return self.get(token) is not None
