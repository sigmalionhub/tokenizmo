"""
Unigram Language Model vocabulary trainer.

Algorithm:
1. Seed candidates: extract all substrings up to max_len from corpus → ~500k items
2. Estimate log-probabilities: log(count / total)
3. EM pruning loop:
   a. For each multi-char token, compute loss_i = expected decrease in corpus
      log-likelihood if the token is removed (approximated as freq * (-log_prob))
   b. Sort by loss ascending (tokens with low loss contribute little)
   c. Remove bottom (1 - shrink_factor) fraction of multi-char tokens
   d. Re-estimate log-probs on remaining vocab
   e. Repeat until |vocab| <= target_size
4. Always preserve all single-character tokens (full Unicode coverage)
5. Always preserve special tokens: <unk>, <s>, </s>, <pad>

Reference: Kudo (2018) "Subword Regularization: Improving Neural Network Translation
Models with Multiple Subword Candidates" https://arxiv.org/abs/1804.10959
"""

from __future__ import annotations

import math
import collections
from typing import Iterable, Iterator

from .vocabulary import Vocabulary
from .trie import Trie
from .viterbi_encoder import ViterbiEncoder


# Special tokens — always assigned IDs 0-3
SPECIAL_TOKENS: list[bytes] = [
    b"<unk>",
    b"<s>",
    b"</s>",
    b"<pad>",
]


class UnigramTrainer:
    """
    Trains a Unigram LM vocabulary via EM pruning.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (default 262,144 = 2^18).
    max_token_len : int
        Maximum byte length of candidate subword tokens.
    shrink_factor : float
        Fraction of multi-char tokens to KEEP per pruning iteration.
    min_freq : int
        Minimum corpus frequency for a candidate to be considered.
    """

    def __init__(
        self,
        vocab_size: int = 262_144,
        max_token_len: int = 32,
        shrink_factor: float = 0.75,
        min_freq: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_token_len = max_token_len
        self.shrink_factor = shrink_factor
        self.min_freq = min_freq

    def train(self, corpus: Iterable[str]) -> Vocabulary:
        """
        Train a vocabulary from a corpus of strings.
        Returns a Vocabulary sorted by log-probability (descending).
        """
        texts = list(corpus)
        print(f"  Seeding candidates from {len(texts)} documents...")
        counts = self._seed_candidates(texts)

        # Guarantee all 128 ASCII bytes are present (count=1 if not seen in corpus).
        for byte_val in range(128):
            tok = bytes([byte_val])
            if tok not in counts:
                counts[tok] = 1

        # Guarantee all single characters from common Unicode ranges so the
        # Viterbi never fails on text that wasn't in the training corpus.
        # Ranges: Cyrillic (U+0400–U+052F), Latin Extended (U+0080–U+024F),
        # Greek (U+0370–U+03FF), CJK Unified (U+4E00–U+9FFF subset).
        _guaranteed_ranges = [
            (0x0080, 0x0250),  # Latin Extended
            (0x0370, 0x0400),  # Greek and Coptic
            (0x0400, 0x0530),  # Cyrillic + Cyrillic Supplement
            (0x4E00, 0x4F00),  # CJK sample (first 256 chars)
        ]
        for start, end in _guaranteed_ranges:
            for cp in range(start, end):
                try:
                    tok = chr(cp).encode("utf-8")
                    if tok not in counts:
                        counts[tok] = 1
                except (UnicodeEncodeError, ValueError):
                    pass

        print(f"  Initial candidates: {len(counts):,}")

        log_probs = _estimate_log_probs(counts)

        # Separate single Unicode characters (always kept) from multi-char candidates.
        # Use Unicode codepoint count, not byte length — Cyrillic/CJK chars are
        # multi-byte UTF-8 but must never be pruned.
        def _is_single_char(tok: bytes) -> bool:
            try:
                return len(tok.decode("utf-8")) == 1
            except UnicodeDecodeError:
                return False

        single_chars = {
            tok: lp for tok, lp in log_probs.items() if _is_single_char(tok)
        }
        multi_tokens = {
            tok: lp for tok, lp in log_probs.items() if not _is_single_char(tok)
        }

        target_multi = max(0, self.vocab_size - len(single_chars) - len(SPECIAL_TOKENS))

        iteration = 0
        while len(multi_tokens) > target_multi:
            iteration += 1
            keep_n = max(target_multi, int(len(multi_tokens) * self.shrink_factor))

            # Build trie for loss estimation
            current_vocab_lp = {**single_chars, **multi_tokens}
            losses = self._compute_losses(multi_tokens, current_vocab_lp, texts[:200])

            # Keep tokens with highest loss (removing them hurts the most)
            sorted_multi = sorted(multi_tokens.items(), key=lambda x: -losses.get(x[0], 0.0))
            multi_tokens = dict(sorted_multi[:keep_n])

            # Re-estimate log probs on remaining vocab
            remaining_counts = {
                tok: counts[tok] for tok in {**single_chars, **multi_tokens}
                if tok in counts
            }
            new_lp = _estimate_log_probs(remaining_counts)
            single_chars = {tok: new_lp.get(tok, lp) for tok, lp in single_chars.items()}
            multi_tokens = {tok: new_lp.get(tok, lp) for tok, lp in multi_tokens.items()}

            total = len(single_chars) + len(multi_tokens) + len(SPECIAL_TOKENS)
            print(f"  Iteration {iteration}: {total:,} tokens remaining (target {self.vocab_size:,})")

            if len(multi_tokens) == 0:
                break

        # Build final vocabulary: special tokens first, then by log-prob descending
        final_lp: dict[bytes, float] = {}
        for tok in SPECIAL_TOKENS:
            final_lp[tok] = 0.0  # special tokens get log_prob=0 (neutral)
        for tok, lp in {**single_chars, **multi_tokens}.items():
            if tok not in final_lp:
                final_lp[tok] = lp

        return Vocabulary.from_log_probs(final_lp)

    @staticmethod
    def _is_single_unicode(tok: bytes) -> bool:
        try:
            return len(tok.decode("utf-8")) == 1
        except UnicodeDecodeError:
            return False

    def _seed_candidates(self, texts: list[str]) -> dict[bytes, int]:
        """Extract substrings up to max_token_len and count frequencies.

        Splits each document into whitespace-delimited tokens, then counts
        all substrings within each token directly into a dict (no intermediate
        list — avoids peak-memory blowup from millions of bytes objects).
        Processes docs in batches and prunes after each batch.
        """
        import re as _re
        BATCH_SIZE = 2_000
        _WORD_RE = _re.compile(rb'\S+')
        counts: dict[bytes, int] = {}
        total = len(texts)
        ml = self.max_token_len

        for batch_start in range(0, total, BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            batch_counts: dict[bytes, int] = {}

            for text in batch:
                data = text.encode("utf-8")
                for m in _WORD_RE.finditer(data):
                    word = m.group()
                    wn = len(word)
                    # Char-start positions: skip UTF-8 continuation bytes.
                    starts = [i for i in range(wn) if not (0x80 <= word[i] <= 0xBF)]
                    ns = len(starts)
                    for si in range(ns):
                        s = starts[si]
                        for ei in range(si + 1, ns + 1):
                            end = starts[ei] if ei < ns else wn
                            if end - s > ml:
                                break
                            tok = word[s:end]
                            # Direct dict increment — avoids Counter overhead.
                            if tok in batch_counts:
                                batch_counts[tok] += 1
                            else:
                                batch_counts[tok] = 1

            # Merge batch into running total.
            for tok, cnt in batch_counts.items():
                if tok in counts:
                    counts[tok] += cnt
                else:
                    counts[tok] = cnt

            # Prune to keep memory bounded.
            if batch_start + BATCH_SIZE < total:
                counts = {
                    tok: cnt for tok, cnt in counts.items()
                    if cnt >= self.min_freq or self._is_single_unicode(tok)
                }

            done = min(batch_start + BATCH_SIZE, total)
            print(f"  Seeding: {done:,}/{total:,} docs, {len(counts):,} candidates")

        # Final filter
        return {
            tok: cnt
            for tok, cnt in counts.items()
            if cnt >= self.min_freq or self._is_single_unicode(tok)
        }

    def _compute_losses(
        self,
        multi_tokens: dict[bytes, float],
        vocab_log_probs: dict[bytes, float],
        sample_texts: list[str],
    ) -> dict[bytes, float]:
        """
        Approximate loss for each token = frequency * (-log_prob).
        Higher loss = removing this token hurts more = keep it.

        This is a frequency-weighted approximation of the full EM loss.
        The exact loss requires re-running Viterbi encoding without the token,
        which is expensive; this approximation works well in practice.
        """
        # Count actual token usage via Viterbi on a sample
        vocab = Vocabulary.from_log_probs(vocab_log_probs)
        encoder = ViterbiEncoder(vocab)

        usage: dict[bytes, int] = collections.Counter()
        for text in sample_texts:
            try:
                ids = encoder.encode(text)
                for tid in ids:
                    base_id = tid & ~(1 << 22)  # strip LEADING_SPACE_FLAG
                    entry = vocab.get_entry(base_id)
                    usage[entry.token] += 1
            except Exception:
                pass

        losses: dict[bytes, float] = {}
        for tok, lp in multi_tokens.items():
            freq = usage.get(tok, 0)
            # Loss = expected log-likelihood contribution of this token
            losses[tok] = freq * (-lp)

        return losses


def _estimate_log_probs(counts: dict[bytes, int]) -> dict[bytes, float]:
    """Estimate log-probabilities from raw counts."""
    total = sum(counts.values())
    if total == 0:
        return {}
    log_total = math.log(total)
    return {
        tok: math.log(cnt) - log_total
        for tok, cnt in counts.items()
        if cnt > 0
    }


def train_from_texts(
    texts: Iterable[str],
    vocab_size: int = 8192,
    **kwargs,
) -> Vocabulary:
    """Convenience function: train a vocabulary from an iterable of strings."""
    trainer = UnigramTrainer(vocab_size=vocab_size, **kwargs)
    return trainer.train(texts)
