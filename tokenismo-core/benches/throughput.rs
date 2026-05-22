use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use std::sync::Arc;
use tokenismo_core::{vocab_io, Encoder, VocabTrie};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_ascii_encoder() -> Encoder {
    let mut trie = VocabTrie::new();
    for b in 0u8..=127 {
        trie.insert(&[b], -(b as f32 * 0.01 + 1.0));
    }
    Encoder::new(Arc::new(trie))
}

fn load_real_encoder() -> Option<Encoder> {
    // Look for the trained vocab relative to the workspace root.
    let candidates = [
        "data/vocab/tokenismo.vocab",
        "../data/vocab/tokenismo.vocab",
        "../../data/vocab/tokenismo.vocab",
    ];
    for p in &candidates {
        if Path::new(p).exists() {
            match vocab_io::load_vocab(Path::new(p)) {
                Ok(trie) => {
                    let enc = Encoder::new(Arc::new(trie));
                    return Some(enc);
                }
                Err(e) => eprintln!("warn: failed to load {p}: {e}"),
            }
        }
    }
    None
}

// ── benchmarks ───────────────────────────────────────────────────────────────

fn bench_ascii_vocab(c: &mut Criterion) {
    let enc = make_ascii_encoder();
    let data = "The quick brown fox jumps over the lazy dog. ".repeat(500);

    let mut group = c.benchmark_group("ascii_vocab");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| enc.encode(std::hint::black_box(&data)))
    });
    group.finish();
}

fn bench_real_vocab(c: &mut Criterion) {
    let Some(enc) = load_real_encoder() else {
        eprintln!("Skipping real-vocab bench: data/vocab/tokenismo.vocab not found");
        return;
    };

    let samples: &[(&str, &str)] = &[
        (
            "en_prose",
            "The tokenizer converts raw text into a sequence of integer token IDs \
             that a language model can process. Good tokenization balances vocabulary \
             coverage, compression ratio, and encoding speed. ",
        ),
        (
            "ru_prose",
            "Токенизатор преобразует исходный текст в последовательность целочисленных \
             идентификаторов токенов, которые может обработать языковая модель. \
             Хорошая токенизация балансирует между покрытием словаря и скоростью. ",
        ),
        (
            "code",
            "def tokenize(text: str) -> list[int]:\n    \
             tokens = []\n    for word in text.split():\n        \
             tokens.extend(encode_word(word))\n    return tokens\n",
        ),
    ];

    for (label, sample) in samples {
        // Build a 64 KB payload by repeating the sample.
        let repeat = (64 * 1024 / sample.len()).max(1);
        let data: String = sample.repeat(repeat);

        let mut group = c.benchmark_group("real_vocab");
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("encode", label), &data, |b, d| {
            b.iter(|| enc.encode(std::hint::black_box(d.as_str())))
        });
        group.finish();
    }
}

fn bench_chunk_sizes(c: &mut Criterion) {
    let Some(enc) = load_real_encoder() else {
        return;
    };

    // Measure how throughput scales with input size (amortises allocation overhead).
    let base = "The tokenizer converts raw text into tokens efficiently. ";
    let mut group = c.benchmark_group("chunk_sizes");

    for size_kb in [1u64, 4, 16, 64, 256] {
        let repeat = ((size_kb as usize * 1024) / base.len()).max(1);
        let data: String = base.repeat(repeat);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("encode_kb", size_kb),
            &data,
            |b, d| b.iter(|| enc.encode(std::hint::black_box(d.as_str()))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ascii_vocab, bench_real_vocab, bench_chunk_sizes);
criterion_main!(benches);
