"""
Baseline benchmark: compare tiktoken and SentencePiece on EN/RU/Code samples.
Establishes the numbers TokeNismo must beat.

Usage:
    python python/benchmarks/baseline.py
"""

import time
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SAMPLES = ROOT / "data" / "samples"


def load_sample(name: str) -> str:
    path = SAMPLES / name
    if not path.exists():
        print(f"  [SKIP] {name} not found — run scripts/fetch_samples.py first")
        return ""
    return path.read_text(encoding="utf-8")


def bench_tokenizer(name: str, encode_fn, texts: dict[str, str]) -> dict:
    results = {}
    for label, text in texts.items():
        if not text:
            continue
        text_bytes = text.encode("utf-8")
        # Warmup
        encode_fn(text[:1000])
        # Timed run
        t0 = time.perf_counter()
        ids = encode_fn(text)
        elapsed = time.perf_counter() - t0

        results[label] = {
            "tokens": len(ids),
            "chars": len(text),
            "bytes": len(text_bytes),
            "chars_per_token": len(text) / len(ids) if ids else 0,
            "bytes_per_token": len(text_bytes) / len(ids) if ids else 0,
            "mb_per_sec": len(text_bytes) / elapsed / 1e6 if elapsed > 0 else 0,
        }
    return results


def print_table(all_results: dict[str, dict]):
    labels = ["sample_en", "sample_ru", "sample_code"]
    metrics = ["tokens", "chars_per_token", "bytes_per_token", "mb_per_sec"]

    print("\n" + "═" * 90)
    print(f"{'Tokenizer':<20} {'Sample':<14} {'Tokens':>8} {'Chars/Tok':>10} {'Bytes/Tok':>10} {'MB/s':>8}")
    print("─" * 90)

    for tok_name, results in all_results.items():
        for label in labels:
            if label not in results:
                continue
            r = results[label]
            print(
                f"{tok_name:<20} {label:<14} {r['tokens']:>8,} "
                f"{r['chars_per_token']:>10.2f} {r['bytes_per_token']:>10.2f} "
                f"{r['mb_per_sec']:>8.1f}"
            )
        print("─" * 90)

    # Cross-lingual parity
    print("\nCross-lingual parity (RU tokens / EN tokens — lower is better):")
    print(f"{'Tokenizer':<20} {'RU/EN ratio':>12}  {'Target: ≤ 1.5'}")
    print("─" * 50)
    for tok_name, results in all_results.items():
        if "sample_en" in results and "sample_ru" in results:
            en_t = results["sample_en"]["tokens"]
            ru_t = results["sample_ru"]["tokens"]
            # Normalize by char count so we compare equivalent-length texts
            en_chars = results["sample_en"]["chars"]
            ru_chars = results["sample_ru"]["chars"]
            ratio = (ru_t / ru_chars) / (en_t / en_chars) if en_t > 0 else 0
            flag = "✅" if ratio <= 1.5 else ("⚠️ " if ratio <= 2.0 else "❌")
            print(f"{tok_name:<20} {ratio:>12.3f}  {flag}")

    print("═" * 90)


def main():
    texts = {
        "sample_en": load_sample("sample_en.txt"),
        "sample_ru": load_sample("sample_ru.txt"),
        "sample_code": load_sample("sample_code.py"),
    }

    if all(not v for v in texts.values()):
        print("No sample files found. Run: python scripts/fetch_samples.py")
        sys.exit(1)

    all_results = {}

    # --- tiktoken ---
    try:
        import tiktoken
        for model_name in ["cl100k_base", "o200k_base"]:
            enc = tiktoken.get_encoding(model_name)
            all_results[f"tiktoken/{model_name}"] = bench_tokenizer(
                model_name, enc.encode, texts
            )
        print("✓ tiktoken benchmarked")
    except ImportError:
        print("✗ tiktoken not installed: pip install tiktoken")

    # --- SentencePiece ---
    try:
        import sentencepiece as spm
        sp_path = ROOT / "data" / "vocab" / "sentencepiece.model"
        if sp_path.exists():
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_path))
            all_results["sentencepiece"] = bench_tokenizer(
                "sentencepiece", sp.EncodeAsIds, texts
            )
            print("✓ SentencePiece benchmarked")
        else:
            print(f"✗ SentencePiece model not found at {sp_path}")
    except ImportError:
        print("✗ sentencepiece not installed: pip install sentencepiece")

    # --- TokeNismo (if built) ---
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from tokenismo import TokeNismo
        for vocab_name, label in [
            ("tokenismo.vocab", "tokenismo/262k"),
            ("tokenismo_small.vocab", "tokenismo/8k"),
        ]:
            vocab_path = ROOT / "data" / "vocab" / vocab_name
            if vocab_path.exists():
                tok = TokeNismo.from_file(str(vocab_path))
                all_results[label] = bench_tokenizer(label, tok.encode, texts)
                print(f"[OK] TokeNismo ({vocab_name}) benchmarked")
    except ImportError:
        print("[!!] TokeNismo not built yet (run maturin develop --release)")

    if not all_results:
        print("No tokenizers available to benchmark.")
        sys.exit(1)

    print_table(all_results)

    # Save results for report generation
    import json
    out_path = ROOT / "Tasks" / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
