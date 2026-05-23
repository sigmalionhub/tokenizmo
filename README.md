# TokeNismo

A next-generation multilingual tokenizer using a Unigram Language Model (ULM)
and Viterbi dynamic-programming segmentation, implemented in Rust with Python
bindings via PyO3.

## Why TokeNismo?

tokenismo v6 (262k vocab) beats every competitor on compression **and** throughput:

### Compression (chars per token — higher is better)

| Tokenizer          | Vocab  |     EN |     RU |   Code | RU/EN ratio |
|:-------------------|-------:|------:|------:|------:|------------:|
| tiktoken cl100k    |   100k |   5.54 |   2.29 |   4.03 |      2.039x |
| tiktoken o200k     |   200k |   5.58 |   4.21 |   4.04 |      1.115x |
| XLM-R (250k)       |   250k |   4.50 |   4.25 |   3.10 |      0.890x |
| mBERT (120k)       |   120k |   4.80 |   3.73 |   2.50 |      1.084x |
| **tokenismo v6**   | **262k** | **5.69** | **5.80** | **4.09** | **0.827x** |

### Throughput (MB/s on 64 KB input — higher is better)

| Tokenizer          |        EN |        RU |      Code |
|:-------------------|----------:|----------:|----------:|
| tiktoken cl100k    |       8.8 |       8.1 |       5.4 |
| tiktoken o200k     |      13.5 |       8.1 |       6.3 |
| **tokenismo v6**   |  **76.0** |  **92.0** |  **55.8** |

### tokenismo v6 vs tiktoken o200k

- **EN**: +2.1% compression, **5.6x** faster
- **RU**: +37.7% compression, **11.4x** faster
- **Code**: +1.3% compression, **8.9x** faster
- **RU/EN ratio**: **0.827x** — Russian is cheaper to tokenize than English

> Benchmarked on 64 KB inputs, Python API (PyO3), warm cache.
> Run `python scripts/benchmark_competitors.py --no-hf` to reproduce.

## Algorithm

```
Corpus → Seed candidates (all substrings ≤ 32 bytes)
       → EM pruning loop (Unigram LM, Viterbi loss estimation)
       → Binary .vocab file (NXT1 format)
       → Rust VocabTrie (flat array-backed, 256-children per node)
       → Viterbi DP encoder (O(n × max_token_len), thread-local DP buffer)
       → Rayon batch encoder (work-stealing, one DP buffer per thread)
```

Key design choices:

- **Viterbi DP** — globally optimal segmentation (minimum tokens, maximum log-prob tiebreak) rather than greedy BPE.
- **Indentation Pre-tokenizer** — multi-space sequences (2, 4, 8 spaces) are isolated into structural chunks before Viterbi decoding. This prevents combinatorial explosion in the DP graph for code files and leverages atomic caching.
- **`LEADING_SPACE_FLAG = 1 << 22`** — leading spaces are encoded as a bit flag on the following token ID in runtime, eliminating standalone space tokens.
- **Thread-local DP buffer** — avoids O(n) heap allocation per `encode()` call; buffer is reused across calls on the same thread.
- **Unicode-aware pruning** — Cyrillic, CJK, and other multi-byte single characters are never pruned from the vocabulary.
- **Graceful OOV handling** — characters outside the vocabulary emit `<unk>` per UTF-8 character; bytes after the gap are re-encoded normally rather than lost.

## Quick Start

### Install

```bash
# Build from source (requires Rust stable + Python 3.9+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install maturin
cd tokenismo-py && maturin develop --release
```

### Encode / Decode

```python
from tokenismo import TokeNismo

tok = TokeNismo.from_file("data/vocab/tokenismo_small.vocab")

# Single text
ids = tok.encode("Hello мир")
text = tok.decode(ids)

# Batch (releases GIL, uses Rayon thread pool)
batch_ids = tok.encode_batch(["Hello", "мир", "world"])
texts = tok.decode_batch(batch_ids)
```

## Training Your Own Vocabulary

```bash
# Small vocabulary for testing (uses bundled sample corpus)
python trainer/train.py \
    --corpus-config configs/corpus_sample.yaml \
    --vocab-size 8192 \
    --output data/vocab/tokenismo_small.vocab

# Full 262k vocabulary (requires Wikipedia EN/RU + The Stack)
# See configs/corpus_full.yaml for download instructions
python trainer/train.py \
    --corpus-config configs/corpus_full.yaml \
    --vocab-size 262144 \
    --output data/vocab/tokenismo.vocab
```

Training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | 262144 | Target vocabulary size |
| `--max-token-len` | 32 | Max byte length of a token candidate |
| `--shrink-factor` | 0.75 | Fraction of multi-char tokens kept per EM iteration |
| `--min-freq` | 2 | Minimum corpus frequency for candidate inclusion |
| `--max-docs` | 0 (all) | Limit documents (useful for quick tests) |

## Benchmarks

See [Why TokeNismo?](#why-tokenismo) above for the full comparison table.

To reproduce:

```bash
python scripts/benchmark_competitors.py        # full benchmark (downloads XLM-R, mBERT)
python scripts/benchmark_competitors.py --no-hf  # tiktoken + tokenismo only
cargo run --release --bin quick_bench           # Rust-level throughput by input size
```

## Vocabulary Format (`.vocab`)

Binary format for fast loading:

```
[magic: 4 bytes "NXT1"]
[vocab_size: u32 LE]
[entry × vocab_size]:
  [token_len: u8]
  [token_bytes: token_len bytes]
  [log_prob: f32 LE]
```

## Development

```bash
# Run all Rust tests (including edge cases and proptest)
cargo test --all --release

# Run property-based tests only (2000 cases by default)
cargo test -p tokenismo-core --test proptest_tests --release

# Run edge case suite
cargo test -p tokenismo-core --test edge_cases --release

# Criterion benchmarks
cargo bench -p tokenismo-core

# Fuzz testing (Linux/macOS, requires nightly)
cargo +nightly fuzz run fuzz_encode -- -max_total_time=3600

# Python integration tests
pytest python/tests/ -v

# Throughput benchmark (requires trained vocab)
python scripts/bench_throughput.py

# Stress test — encode/decode round-trip on 58 extreme cases
python scripts/run_stress_test.py
```

## Project Structure

```
tokenismo-core/       Rust library: VocabTrie, Encoder, Decoder, batch
tokenismo-py/         PyO3 bindings: TokeNismo Python class
trainer/              Python training pipeline: Unigram LM + Viterbi
  unigram_trainer.py  EM pruning loop
  viterbi_encoder.py  Reference Viterbi implementation
  corpus.py           Weighted corpus streaming
  normalizer.py       Unicode normalization (NFC, zero-width strip)
python/               Python package root
  tokenismo/          Public package (wraps Rust extension)
  benchmarks/         Compression benchmarks vs tiktoken
  tests/              Python integration tests
configs/              Corpus YAML configs (sample + full)
data/                 Vocab files and sample corpus
scripts/              Utility scripts (bench, stress test, corpus download)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
