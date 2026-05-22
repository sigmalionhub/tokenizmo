"""
Viterbi encoder: finds the minimum-token-count segmentation of input text.

This is the Python reference implementation. The Rust core (TASK-03) must
produce identical output for all inputs.

Algorithm (Viterbi DP over UTF-8 bytes):
    dp[i] = best (token_count, neg_log_prob_sum) to encode text[0:i]
    For each position i:
        Walk trie from i, collecting all matching tokens ending at j.
        Update dp[j] if dp[i] + 1 token is better than current dp[j].
    Backtrack from dp[n] to recover the token sequence.

Leading-space optimization:
    If byte at position i is a space and the next position has tokens,
    try encoding those tokens with LEADING_SPACE_FLAG instead of emitting
    the space as a standalone token. Both options are tried; DP picks the best.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

from .trie import Trie
from .vocabulary import Vocabulary

LEADING_SPACE_FLAG: int = 1 << 22


class _DpState(NamedTuple):
    count: int          # number of tokens to reach this position
    neg_lp: float       # -sum(log_probs) — lower means higher total probability
    prev_pos: int       # position of the previous token's start
    token_id: int       # token ID used to reach this position (-1 for start)


_UNREACHABLE = _DpState(count=10 ** 9, neg_lp=math.inf, prev_pos=-1, token_id=-1)
_START = _DpState(count=0, neg_lp=0.0, prev_pos=-1, token_id=-1)


def _better(a: _DpState, b: _DpState) -> bool:
    """Returns True if `a` is a better DP state than `b`."""
    return (a.count, a.neg_lp) < (b.count, b.neg_lp)


class ViterbiEncoder:
    """
    Encodes text into a token ID sequence minimizing total token count.
    Tiebreak: maximize total log-probability.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        self._trie = Trie()
        for entry in vocab:
            self._trie.insert(entry.token, entry.log_prob)

    def encode(self, text: str) -> list[int]:
        """
        Encode `text` into a list of token IDs.
        Tokens preceded by a space in the original text have LEADING_SPACE_FLAG set.
        """
        data: bytes = text.encode("utf-8")
        n = len(data)

        if n == 0:
            return []

        dp: list[_DpState] = [_UNREACHABLE] * (n + 1)
        dp[0] = _START

        for i in range(n):
            if dp[i] is _UNREACHABLE or dp[i].count == _UNREACHABLE.count:
                continue

            if data[i] == ord(" ") and i + 1 < n:
                # Try tokens starting at i+1 with LEADING_SPACE_FLAG
                for end, tid, lp in self._trie.walk(data, i + 1):
                    flagged_tid = tid | LEADING_SPACE_FLAG
                    candidate = _DpState(
                        count=dp[i].count + 1,
                        neg_lp=dp[i].neg_lp - lp,
                        prev_pos=i,
                        token_id=flagged_tid,
                    )
                    if _better(candidate, dp[end]):
                        dp[end] = candidate

                # Also try the space itself as a standalone token
                for end, tid, lp in self._trie.walk(data, i):
                    candidate = _DpState(
                        count=dp[i].count + 1,
                        neg_lp=dp[i].neg_lp - lp,
                        prev_pos=i,
                        token_id=tid,
                    )
                    if _better(candidate, dp[end]):
                        dp[end] = candidate
            else:
                for end, tid, lp in self._trie.walk(data, i):
                    candidate = _DpState(
                        count=dp[i].count + 1,
                        neg_lp=dp[i].neg_lp - lp,
                        prev_pos=i,
                        token_id=tid,
                    )
                    if _better(candidate, dp[end]):
                        dp[end] = candidate

        if dp[n].count == _UNREACHABLE.count:
            # Byte fallback: emit each byte as its own token ID.
            # This can only happen if the vocab does not cover some byte values,
            # which should not occur with a properly trained vocab.
            # Use actual token IDs by looking up each byte in the trie;
            # if absent, raise so the caller can diagnose the vocab gap.
            ids_out = []
            for byte in data:
                tid = self._trie.get(bytes([byte]))
                if tid is None:
                    raise ValueError(
                        f"Byte fallback failed: byte 0x{byte:02x} has no token in vocab. "
                        "Ensure vocab covers all single bytes/characters."
                    )
                ids_out.append(tid)
            return ids_out

        return _backtrack(dp, n)

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for tid in ids:
            has_space = bool(tid & LEADING_SPACE_FLAG)
            base_id = tid & ~LEADING_SPACE_FLAG
            entry = self._vocab.get_entry(base_id)
            if has_space:
                buf.append(ord(" "))
            buf.extend(entry.token)
        return buf.decode("utf-8")


def _backtrack(dp: list[_DpState], n: int) -> list[int]:
    tokens: list[int] = []
    pos = n
    while pos > 0:
        state = dp[pos]
        tokens.append(state.token_id)
        pos = state.prev_pos
    tokens.reverse()
    return tokens
