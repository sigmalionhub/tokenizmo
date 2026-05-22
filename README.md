# TokeNismo

A next-generation multilingual tokenizer using a Unigram Language Model (ULM)
and Viterbi dynamic-programming segmentation, implemented in Rust with Python
bindings via PyO3.

## Why TokeNismo?

| Metric | OpenAI o200k\_base | TokeNismo (8k vocab) |
|--------|-------------------|----------------------|
| RU/EN token ratio | 1.324× | **1.199×** |
| Russian vs o200k | 1.0× (baseline) | **0.95×** (fewer tokens) |
| Code vs o200k | 1.0× (baseline) | **0.52×** (2× fewer tokens) |

Lower ratio = fewer tokens = cheaper API calls for Russian and code-heavy workloads.

With a 262k-token vocabulary the RU/EN ratio is expected to drop below **1.1×**,
approaching parity.

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

- **Viterbi DP** — globally optimal segmentation (minimum tokens, maximum log-prob tiebreak) rather than greedy BPE
- **`LEADING_SPACE_FLAG = 1 << 22`** — leading spaces are encoded as a bit flag on the following token, eliminating standalone space tokens
- **Thread-local DP buffer** — avoids O(n) heap allocation per `encode()` call; buffer is reused across calls on the same thread
- **Unicode-aware pruning** — Cyrillic, CJK, and other multi-byte single characters are never pruned from the vocabulary

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

### Throughput (Rust encoder, single thread)

Measured on a minimal 8k vocab with ASCII-only input:

```
Throughput (best of 5): 14 MB/s
Input: 6,400 KB -> 1,220,000 tokens
```

> The 500 MB/s target requires a compact double-array trie (currently flat
> 256-children array). This optimization is tracked in TASK-03.

### Compression vs tiktoken o200k\_base (8k vocab, sample corpus)

```
English text:   TokeNismo 1.05×  (slightly more tokens — expected at 8k vocab)
Russian text:   TokeNismo 0.95×  (5% fewer tokens than o200k)
Python code:    TokeNismo 0.52×  (2× fewer tokens than o200k)
RU/EN ratio:    TokeNismo 1.20×  (target ≤ 1.5 — ACHIEVED)
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
scripts/              Utility scripts (bench, wiki converter)
Tasks/                Project roadmap and task files
```

## License

MIT
