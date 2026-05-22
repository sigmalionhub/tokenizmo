# TokeNismo Architecture

## Overview

```
                 Training Pipeline (Python)
                 ┌──────────────────────────┐
  Corpus text ──►│  UnigramTrainer          │
                 │  1. Seed candidates      │
                 │     (all substrings ≤32B)│
                 │  2. EM pruning loop      │
                 │     Viterbi loss estimate│
                 │  3. Final vocab          │
                 └───────────┬──────────────┘
                             │ .vocab binary (NXT1)
                             ▼
                 ┌──────────────────────────┐
                 │  VocabTrie (Rust)        │
                 │  Flat array-backed trie  │
                 │  256 children per node   │
                 └───────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────┐         ┌──────────────────┐
   │ Encoder (Rust)   │         │ Decoder (Rust)   │
   │ Viterbi DP       │         │ id → bytes → str │
   │ LEADING_SPACE    │         │ LEADING_SPACE     │
   │ thread-local buf │         │ flag handling     │
   └────────┬─────────┘         └──────────────────┘
            │ Rayon batch
            ▼
   ┌──────────────────┐
   │ PyO3 bindings    │
   │ TokeNismo class  │
   │ GIL release      │
   └──────────────────┘
```

## Vocabulary Format

```
Offset  Size  Field
0       4     Magic: "NXT1"
4       4     vocab_size (u32 LE)
8+      var   Entries (vocab_size times):
              [token_len: u8][token_bytes: N bytes][log_prob: f32 LE]
```

Token IDs are assigned in order of insertion (0, 1, 2, ...).  Special tokens
`<unk>`, `<s>`, `</s>`, `<pad>` always occupy IDs 0–3.

## VocabTrie

A flat array-backed trie.  Node 0 is the root.  Each node has 256 child slots
(one per byte value), stored contiguously:

```
children[node * 256 + byte] = child_node  (0 = no child)
token_ids[node]              = token_id   (u32::MAX = not a terminal)
```

This layout gives O(1) child lookup with good cache locality for sequential
byte scans.

## Viterbi Encoder

For input of length `n` bytes:

1. Allocate (or reuse thread-local) DP buffer of size `n+1`.
2. For each position `i`:
   - Walk the trie from `i`, collecting all matching tokens ending at each `j > i`.
   - If `bytes[i] == ' '` and `i+1 < n`, try matching tokens starting at `i+1`
     with `LEADING_SPACE_FLAG` set (absorbs the space into the next token).
   - Update `dp[j]` if the candidate is better (fewer tokens; ties broken by
     higher total log-probability).
3. Backtrack from `dp[n]` to reconstruct the token sequence.

**Complexity:** O(n × max_token_len) time, O(n) space.

## LEADING_SPACE_FLAG

```
Bit 22 of a token ID = the token was preceded by a space in the original text.
```

Encoding `"hello world"`:
- Without flag:  `["hello", " ", "world"]` — 3 tokens
- With flag:     `["hello", " world"]`     — 2 tokens (space absorbed)

The decoder checks bit 22: if set, prepend `b' '` before the token bytes.

## Batch Encoding

`encode_batch` uses Rayon's work-stealing thread pool.  Each worker thread
maintains its own `thread_local!` DP buffer, eliminating lock contention and
heap allocation in the hot path.

The Python `encode_batch` method calls `py.allow_threads()` to release the GIL
during Rayon execution, enabling true parallel encoding from multiple Python
threads.

## Training: Unigram EM Loop

```
1. Seed: extract all byte substrings ≤ max_token_len from corpus.
   Assign initial log_prob = log(count / total).
   Guarantee all 128 ASCII bytes are present.
   Guarantee all single Unicode characters are present (by char count, not byte count).

2. Loop until |multi_char_tokens| ≤ target:
   a. Estimate token usage via Viterbi on a sample (200 documents).
   b. loss(token) = usage_count × (−log_prob)
      Higher loss = removing this token hurts more.
   c. Keep top shrink_factor fraction by loss.
   d. Re-estimate log_probs from raw counts on remaining vocab.

3. Output: special tokens (IDs 0–3) + single chars + multi-char tokens,
   sorted by log_prob descending.
```

Key invariants:
- Single Unicode characters (Cyrillic, CJK, etc.) are **never pruned**.
- All 128 ASCII bytes are always present.
- These two rules guarantee the Viterbi always has a complete-coverage fallback.
