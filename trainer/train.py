"""
TokeNismo vocabulary trainer entry point.

Trains a Unigram LM vocabulary from a multilingual corpus and saves it
in the binary .vocab format readable by the Rust core.

Usage:
    python trainer/train.py --corpus-config configs/corpus_sample.yaml \\
        --vocab-size 8192 --output data/vocab/tokenismo_small.vocab

    python trainer/train.py --corpus-config configs/corpus_full.yaml \\
        --vocab-size 262144 --output data/vocab/tokenismo.vocab --seed 42
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TokeNismo vocabulary")
    p.add_argument("--corpus-config", required=True, help="YAML corpus config file")
    p.add_argument("--vocab-size", type=int, default=262_144)
    p.add_argument("--output", required=True, help="Output .vocab binary file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-token-len", type=int, default=32)
    p.add_argument("--shrink-factor", type=float, default=0.75)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-docs", type=int, default=0,
                   help="Limit documents for testing (0 = no limit)")
    return p.parse_args()


def load_corpus_config(config_path: Path):
    """Load corpus config YAML and return a CorpusStream."""
    try:
        import yaml
    except ImportError:
        # Fallback: minimal YAML parser for our simple config format
        return _load_config_fallback(config_path)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _build_stream(cfg)


def _load_config_fallback(config_path: Path):
    """Parse our simple YAML config without pyyaml dependency."""
    from trainer.corpus import CorpusSource, CorpusStream

    # Very basic parser: only handles our specific config structure
    sources = []
    mix_ratios = {}
    current_source: dict = {}
    in_sources = False
    in_mix = False

    with open(config_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("mix_ratios:"):
                in_mix = True
                in_sources = False
                continue
            if stripped.startswith("sources:"):
                in_sources = True
                in_mix = False
                if current_source:
                    sources.append(current_source)
                    current_source = {}
                continue

            if in_mix and ":" in stripped:
                k, v = stripped.split(":", 1)
                mix_ratios[k.strip()] = float(v.strip())

            if in_sources:
                if stripped.startswith("- path:"):
                    if current_source:
                        sources.append(current_source)
                    current_source = {"path": stripped[7:].strip()}
                elif ":" in stripped and current_source:
                    k, v = stripped.split(":", 1)
                    current_source[k.strip()] = v.strip()

    if current_source:
        sources.append(current_source)

    corpus_sources = []
    for s in sources:
        if "path" not in s:
            continue
        path = ROOT / s["path"]
        corpus_sources.append(CorpusSource(
            path=path,
            language=s.get("language", "en"),
            weight=mix_ratios.get(s.get("language", "en"), 1.0),
            reader=s.get("reader", "text"),
            text_field=s.get("text_field", "text"),
        ))

    return CorpusStream(corpus_sources, mix_ratios=mix_ratios)


def _build_stream(cfg: dict):
    from trainer.corpus import CorpusSource, CorpusStream
    mix_ratios = cfg.get("mix_ratios", {})
    sources = []
    for s in cfg.get("sources", []):
        path = ROOT / s["path"]
        sources.append(CorpusSource(
            path=path,
            language=s.get("language", "en"),
            weight=mix_ratios.get(s.get("language", "en"), 1.0),
            reader=s.get("reader", "text"),
            text_field=s.get("text_field", "text"),
        ))
    return CorpusStream(sources, mix_ratios=mix_ratios)


def save_vocab_binary(vocab, output_path: Path) -> None:
    """
    Write vocabulary to binary .vocab file.
    Format: [magic:4][vocab_size:4u32LE][entries...]
    Entry:  [token_len:1u8][token_bytes:N][log_prob:4f32LE]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"NXT1")
        entries = list(vocab)
        f.write(struct.pack("<I", len(entries)))
        for entry in entries:
            tok = entry.token
            assert len(tok) <= 255, f"Token too long: {tok!r}"
            f.write(struct.pack("<B", len(tok)))
            f.write(tok)
            f.write(struct.pack("<f", entry.log_prob))
    print(f"  Saved {len(entries):,} tokens to {output_path}")


def validate_vocab(vocab, corpus_sample: list[str]) -> None:
    """Quick sanity checks on the trained vocabulary."""
    from trainer.viterbi_encoder import ViterbiEncoder

    tokens = {e.token for e in vocab}
    # All printable ASCII must be present
    missing_ascii = [
        chr(i) for i in range(32, 127)
        if chr(i).encode() not in tokens
    ]
    if missing_ascii:
        print(f"  WARNING: {len(missing_ascii)} ASCII chars missing from vocab: "
              f"{''.join(missing_ascii[:20])}")
    else:
        print(f"  [OK] All printable ASCII chars present")

    # Round-trip test on sample
    encoder = ViterbiEncoder(vocab)
    failures = 0
    unk_bytes = 0
    total_ids = 0
    for text in corpus_sample[:20]:
        try:
            ids = encoder.encode(text)
            recovered = encoder.decode(ids)
            total_ids += len(ids)
            if recovered != text:
                failures += 1
        except ValueError as e:
            unk_bytes += 1

    print(f"  [OK] Round-trip: {len(corpus_sample[:20]) - failures}/20 texts lossless")
    if total_ids > 0:
        avg_chars = sum(len(t) for t in corpus_sample[:20]) / total_ids
        print(f"  Avg chars/token on sample: {avg_chars:.2f}")


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    print("=" * 60)
    print(f"TokeNismo Vocabulary Trainer")
    print(f"  vocab_size = {args.vocab_size:,}")
    print(f"  output     = {args.output}")
    print(f"  config     = {args.corpus_config}")
    print("=" * 60)

    # Load corpus
    print("\nLoading corpus config...")
    stream = load_corpus_config(Path(args.corpus_config))
    est_bytes = stream.estimate_size()
    print(f"  Corpus sources: {len(stream.sources)}")
    if est_bytes:
        print(f"  Estimated size: {est_bytes / 1e6:.1f} MB")

    # Collect documents
    print("\nCollecting documents...")
    docs: list[str] = []
    from trainer.normalizer import normalize
    for i, doc in enumerate(stream):
        docs.append(normalize(doc))
        if args.max_docs and i + 1 >= args.max_docs:
            break
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} docs loaded...", end="\r")
    print(f"  Collected {len(docs):,} documents, "
          f"{sum(len(d) for d in docs):,} total chars")

    if not docs:
        print("ERROR: No documents loaded. Check corpus config and file paths.")
        sys.exit(1)

    # Train vocabulary
    print(f"\nTraining vocabulary (target size: {args.vocab_size:,})...")
    from trainer.unigram_trainer import UnigramTrainer
    trainer = UnigramTrainer(
        vocab_size=args.vocab_size,
        max_token_len=args.max_token_len,
        shrink_factor=args.shrink_factor,
        min_freq=args.min_freq,
    )
    vocab = trainer.train(docs)
    print(f"  Final vocabulary: {len(vocab):,} tokens")

    # Validate
    print("\nValidating vocabulary...")
    validate_vocab(vocab, docs)

    # Save
    print("\nSaving vocabulary...")
    save_vocab_binary(vocab, Path(args.output))

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
