# TokeNismo — Project Documentation

Next-generation multilingual tokenizer targeting superior compression and multi-GB/s throughput.
Beats OpenAI `o200k_base` on Russian/English parity (target: ≤ 1.5x token ratio) and speed (target: ≥ 2 GB/s).

## Status

| Phase | Task | Status |
|-------|------|--------|
| 0 Setup | TASK-00 | ✅ Done |
| 1 Prototype | TASK-01 | ✅ Done |
| 2 Training | TASK-02 | ✅ Done |
| 3 Rust Core | TASK-03 | ✅ Done |
| 4 Bindings | TASK-04 | ✅ Done |
| 5 Benchmarks | TASK-05 | ✅ Done |
| 6 Hardening | TASK-06 | ✅ Done |

---

## Architecture

```
tokenismo/
├── tokenismo-core/          Rust lib — VocabTrie, Encoder, Decoder, batch
├── tokenismo-py/            PyO3 bindings crate → Python package
├── python/
│   ├── tokenismo/           Python package (__init__.py re-exports TokeNismo)
│   └── benchmarks/          Benchmark scripts (baseline.py, etc.)
├── trainer/                 Python vocab training scripts
│   └── train.py             Entry point (stub; full impl in TASK-02)
├── scripts/
│   └── fetch_samples.py     Download/generate sample data files
├── data/
│   ├── samples/             EN/RU/Code benchmark samples (~15KB each)
│   └── vocab/               Trained vocab files (.vocab binary)
└── Tasks/                   Task specs, roadmap, results
```

**Data flow:**
```
raw text
  → normalizer (NFC, whitespace)
  → VocabTrie prefix scan
  → Viterbi DP (minimize token count)
  → token ID sequence (with LEADING_SPACE_FLAG on applicable tokens)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `tokenismo-core/src/vocab.rs` | Array-backed trie for O(L) prefix lookup |
| `tokenismo-core/src/encoder.rs` | Viterbi DP encoder, `LEADING_SPACE_FLAG` |
| `tokenismo-core/src/decoder.rs` | Token ID → UTF-8 decoder |
| `tokenismo-core/src/batch.rs` | Rayon parallel batch encoding |
| `tokenismo-core/src/vocab_io.rs` | Binary `.vocab` file save/load |
| `tokenismo-py/src/lib.rs` | PyO3 `TokeNismo` Python class |
| `python/tokenismo/__init__.py` | Python package entry point |
| `python/benchmarks/baseline.py` | Baseline comparison vs tiktoken/SP |
| `data/samples/` | Sample EN/RU/Code files for benchmarking |
| `Tasks/baseline_results.md` | Baseline numbers (populate by running benchmark) |
| `trainer/__init__.py` | Python trainer package exports |
| `trainer/vocabulary.py` | `Vocabulary` + `VocabEntry` data structures |
| `trainer/trie.py` | Byte-level `Trie` with `.walk()` for prefix scan |
| `trainer/normalizer.py` | Unicode NFC normalization + `WhitespaceHandler` |
| `trainer/viterbi_encoder.py` | **Reference Viterbi encoder** (Python, must match Rust output) |
| `trainer/unigram_trainer.py` | `UnigramTrainer` — EM pruning from 500k candidates |
| `trainer/test_prototype.py` | Integration tests: round-trip, parity, count comparison |

---

## Public API

### Rust (`tokenismo-core`)

```rust
// Vocabulary trie
let mut trie = VocabTrie::new();
trie.insert(b"hello", -1.0_f32);  // (token_bytes, log_prob) → token_id
trie.get(b"hello");                // → Option<u32>

// Encoder
let enc = Encoder::new(Arc::new(trie));
enc.encode("hello world")         // → Vec<u32>  (allocates buffers internally)
enc.encode_into(text, &mut dp_buf, &mut out)  // zero-alloc hot path

// Batch (Rayon parallel)
encode_batch(&enc, &["hello", "world"])  // → Vec<Vec<u32>>

// Decoder
let dec = Decoder::from_trie(&trie);
dec.decode(&ids)                  // → Result<String, DecodeError>

// Vocab I/O
vocab_io::save_vocab(&trie, Path::new("out.vocab"))
vocab_io::load_vocab(Path::new("out.vocab"))  // → Result<VocabTrie, VocabError>
```

**`LEADING_SPACE_FLAG = 1 << 22`** — set on token ID when preceded by a space. The decoder
inserts a space before the token's bytes when this flag is present.

### Python (`tokenismo`)

```python
from tokenismo import TokeNismo

tok = TokeNismo.from_file("data/vocab/tokenismo.vocab")
tok.encode("hello мир")          # → list[int]
tok.encode_batch(["a", "b"])     # → list[list[int]]  (GIL released)
tok.decode([1, 2, 3])            # → str
tok.decode_batch([[1,2],[3,4]])   # → list[str]  (GIL released)
tok.vocab_size                   # → int
```

---

## Development Commands

```bash
# Fetch sample data
python scripts/fetch_samples.py

# Run baseline benchmarks (requires tiktoken: pip install tiktoken)
python python/benchmarks/baseline.py

# Build Python extension (requires Rust + maturin)
cd tokenismo-py
maturin develop --release

# Run Rust tests
cargo test -p tokenismo-core

# Run Rust benchmarks
cargo bench -p tokenismo-core

# Check compilation
cargo check
```

---

## Vocab File Format (`.vocab`)

Binary format produced by `vocab_io::save_vocab`:
```
[magic: 4 bytes "NXT1"]
[vocab_size: u32 LE]
[per token: token_len:u8 | token_bytes:N | log_prob:f32 LE]
```
Tokens are stored in insertion order (which equals token ID order).

---

## Python Trainer API

```python
from trainer import UnigramTrainer, ViterbiEncoder, normalize, Vocabulary

# Normalize text before training/encoding
clean = normalize(raw_text)  # NFC, strip zero-width chars, canonicalize newlines

# Train a vocabulary
trainer = UnigramTrainer(vocab_size=262_144, max_token_len=32, shrink_factor=0.75)
vocab = trainer.train(["doc1 text", "doc2 text", ...])  # iterable of strings

# Encode / decode
encoder = ViterbiEncoder(vocab)
ids = encoder.encode("hello мир")   # → list[int]
text = encoder.decode(ids)          # → str (lossless round-trip)

# Run integration tests
# python -m trainer.test_prototype
```

**LEADING_SPACE_FLAG = `1 << 22`** — same constant in Python and Rust.
Tokens with this flag set in their ID were preceded by a space; the decoder
inserts a `' '` before their bytes.

**Viterbi optimality guarantee:** `encode()` returns the token sequence with
the fewest tokens. On a tie, picks the sequence with highest total log-probability.

## Known Limitations

- **Throughput: 14 MB/s** (target ≥ 500 MB/s) — the flat 256-children trie has poor cache behaviour for large vocabularies. Requires a compact double-array trie or SIMD-accelerated implementation. Tracked as a future TASK-03 optimization.
- **Full 262k vocab** requires downloading Wikipedia EN/RU dumps + The Stack (~100 GB). See `configs/corpus_full.yaml` for instructions.
- **Fuzz testing** (`tokenismo-core/fuzz/`) requires `cargo +nightly fuzz run fuzz_encode` on Linux or macOS. Not supported on Windows (libFuzzer limitation).
- **Wheel publishing** (PyPI/crates.io) not yet automated — requires a GitHub Actions release workflow.

## Testing

```bash
# All Rust tests (unit + edge cases + proptest)
cargo test -p tokenismo-core --release

# Edge cases only
cargo test -p tokenismo-core --test edge_cases --release

# Proptest only (2000 generated cases per property)
cargo test -p tokenismo-core --test proptest_tests --release

# Fuzz (Linux/macOS, nightly required)
cargo +nightly fuzz run fuzz_encode -- -max_total_time=3600

# Python integration tests
python -m pytest python/tests/ -v

# Throughput benchmark (requires trained vocab)
python scripts/bench_throughput.py
```
