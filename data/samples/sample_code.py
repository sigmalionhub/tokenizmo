#!/usr/bin/env python3
"""
Sample Python code for tokenizer benchmarking.
Implements a simplified Unigram LM trainer and Viterbi encoder.
"""

from __future__ import annotations

import math
import heapq
import collections
from typing import Iterator, Optional, NamedTuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    text: str
    log_prob: float
    token_id: int

    def __lt__(self, other: Token) -> bool:
        return self.log_prob > other.log_prob


class TrieNode:
    __slots__ = ("children", "token_id", "log_prob")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.token_id: Optional[int] = None
        self.log_prob: float = float("-inf")


class VocabTrie:
    """Trie-based vocabulary for O(L) prefix lookups."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0
        self._tokens: list[tuple[bytes, float]] = []

    def insert(self, token: str | bytes, log_prob: float) -> int:
        raw = token.encode("utf-8") if isinstance(token, str) else token
        node = self.root
        for byte in raw:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        token_id = self._size
        node.token_id = token_id
        node.log_prob = log_prob
        self._tokens.append((raw, log_prob))
        self._size += 1
        return token_id

    def get(self, token: str | bytes) -> Optional[int]:
        raw = token.encode("utf-8") if isinstance(token, str) else token
        node = self.root
        for byte in raw:
            node = node.children.get(byte)
            if node is None:
                return None
        return node.token_id

    def __len__(self) -> int:
        return self._size

    def __contains__(self, token: str | bytes) -> bool:
        return self.get(token) is not None


# ---------------------------------------------------------------------------
# Viterbi encoder
# ---------------------------------------------------------------------------

class DpState(NamedTuple):
    count: int           # number of tokens
    neg_log_prob: float  # -sum(log_probs) — lower is better
    prev_pos: int
    token_id: int


_UNREACHABLE = DpState(count=10**9, neg_log_prob=float("inf"), prev_pos=-1, token_id=-1)

LEADING_SPACE_FLAG = 1 << 22


def viterbi_encode(text: str, trie: VocabTrie) -> list[int]:
    """
    Encode `text` into the minimum-token-count sequence using Viterbi DP.

    Returns a list of token IDs. Tokens preceded by a space have
    LEADING_SPACE_FLAG set in their ID.
    """
    data = text.encode("utf-8")
    n = len(data)
    dp: list[DpState] = [_UNREACHABLE] * (n + 1)
    dp[0] = DpState(count=0, neg_log_prob=0.0, prev_pos=-1, token_id=-1)

    for i in range(n):
        if dp[i].count == _UNREACHABLE.count:
            continue

        # Try with leading-space absorption
        if data[i] == ord(" ") and i + 1 < n:
            _scan(data, i + 1, n, i, dp, trie, leading_space=True)
            # Also try space as standalone token
            _match_at(data, i, i + 1, i, dp, trie, leading_space=False)
        else:
            _scan(data, i, n, i, dp, trie, leading_space=False)

    if dp[n].count == _UNREACHABLE.count:
        # Byte fallback: one token per byte
        return list(data)

    return _backtrack(dp, n)


def _scan(
    data: bytes,
    start: int,
    n: int,
    origin: int,
    dp: list[DpState],
    trie: VocabTrie,
    leading_space: bool,
) -> None:
    node = trie.root
    for j in range(start, n):
        node = node.children.get(data[j])
        if node is None:
            break
        if node.token_id is not None:
            tid = node.token_id | (LEADING_SPACE_FLAG if leading_space else 0)
            _update_dp(dp, origin, j + 1, tid, node.log_prob)


def _match_at(
    data: bytes,
    start: int,
    end: int,
    origin: int,
    dp: list[DpState],
    trie: VocabTrie,
    leading_space: bool,
) -> None:
    node = trie.root
    for b in data[start:end]:
        node = node.children.get(b)
        if node is None:
            return
    if node.token_id is not None:
        tid = node.token_id | (LEADING_SPACE_FLAG if leading_space else 0)
        _update_dp(dp, origin, end, tid, node.log_prob)


def _update_dp(
    dp: list[DpState],
    origin: int,
    end: int,
    token_id: int,
    log_prob: float,
) -> None:
    candidate = DpState(
        count=dp[origin].count + 1,
        neg_log_prob=dp[origin].neg_log_prob - log_prob,
        prev_pos=origin,
        token_id=token_id,
    )
    if (candidate.count, candidate.neg_log_prob) < (dp[end].count, dp[end].neg_log_prob):
        dp[end] = candidate


def _backtrack(dp: list[DpState], n: int) -> list[int]:
    tokens: list[int] = []
    pos = n
    while pos > 0:
        state = dp[pos]
        tokens.append(state.token_id)
        pos = state.prev_pos
    tokens.reverse()
    return tokens


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode(token_ids: list[int], vocab_trie: VocabTrie) -> str:
    buf = bytearray()
    for tid in token_ids:
        has_space = bool(tid & LEADING_SPACE_FLAG)
        base_id = tid & ~LEADING_SPACE_FLAG
        if base_id >= len(vocab_trie._tokens):
            raise ValueError(f"Unknown token id: {base_id}")
        raw, _ = vocab_trie._tokens[base_id]
        if has_space:
            buf.append(ord(" "))
        buf.extend(raw)
    return buf.decode("utf-8")


# ---------------------------------------------------------------------------
# Unigram LM trainer (simplified)
# ---------------------------------------------------------------------------

def sliding_substrings(text: str, max_len: int = 16) -> Iterator[str]:
    n = len(text)
    for i in range(n):
        for length in range(1, min(max_len + 1, n - i + 1)):
            yield text[i : i + length]


def estimate_log_probs(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    log_total = math.log(total)
    return {tok: math.log(cnt) - log_total for tok, cnt in counts.items()}


def compute_token_loss(
    token: str,
    vocab_log_probs: dict[str, float],
    corpus_sample: list[str],
    trie: VocabTrie,
) -> float:
    """Estimate how much corpus log-likelihood drops if we remove `token`."""
    lp = vocab_log_probs.get(token, 0.0)
    freq = sum(text.count(token) for text in corpus_sample[:100])
    return freq * (-lp)  # simplified: actual EM uses full Viterbi re-encoding


def prune_vocabulary(
    log_probs: dict[str, float],
    corpus_sample: list[str],
    target_size: int,
    shrink_factor: float = 0.75,
) -> dict[str, float]:
    """Iteratively prune vocabulary to target_size using entropy-based loss."""
    # Always keep single-character tokens
    single_chars = {tok: lp for tok, lp in log_probs.items() if len(tok) == 1}
    multi_tokens = {tok: lp for tok, lp in log_probs.items() if len(tok) > 1}

    trie = VocabTrie()
    for tok, lp in log_probs.items():
        trie.insert(tok, lp)

    while len(single_chars) + len(multi_tokens) > target_size:
        losses = {
            tok: compute_token_loss(tok, log_probs, corpus_sample, trie)
            for tok in multi_tokens
        }
        keep_n = max(len(single_chars), int(len(multi_tokens) * shrink_factor))
        sorted_multi = sorted(multi_tokens.items(), key=lambda x: -losses.get(x[0], 0))
        multi_tokens = dict(sorted_multi[:keep_n])

        if not multi_tokens:
            break

    return {**single_chars, **multi_tokens}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_small_vocab(corpus: list[str], vocab_size: int = 512) -> VocabTrie:
    """Train a small vocabulary for testing."""
    counts: dict[str, int] = collections.Counter()
    for text in corpus:
        for substr in sliding_substrings(text, max_len=8):
            counts[substr] += 1

    log_probs = estimate_log_probs(dict(counts))
    pruned = prune_vocabulary(log_probs, corpus, target_size=vocab_size)

    trie = VocabTrie()
    for token, lp in sorted(pruned.items(), key=lambda x: -x[1]):
        trie.insert(token, lp)
    return trie


def roundtrip_test(text: str, trie: VocabTrie) -> bool:
    ids = viterbi_encode(text, trie)
    recovered = decode(ids, trie)
    return recovered == text


if __name__ == "__main__":
    corpus = [
        "hello world, this is a test",
        "tokenization is important for NLP",
        "the quick brown fox jumps over the lazy dog",
        "Привет мир, это тест токенизации",
        "for i in range(n): print(f'token {i}')",
    ]

    print("Training small vocabulary...")
    trie = train_small_vocab(corpus, vocab_size=256)
    print(f"Vocabulary size: {len(trie)}")

    for text in corpus:
        ids = viterbi_encode(text, trie)
        ok = roundtrip_test(text, trie)
        print(f"  [{len(ids):3d} tokens] {'✓' if ok else '✗'} {text[:50]}")
