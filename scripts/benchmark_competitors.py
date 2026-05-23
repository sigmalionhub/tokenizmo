"""
Comprehensive benchmark: tokenismo v6 vs competitor tokenizers.

Competitors:
  tiktoken   cl100k_base  (GPT-3.5/4,    BPE  100k vocab)
  tiktoken   o200k_base   (GPT-4o,        BPE  200k vocab)
  XLM-R      xlm-roberta  (SentencePiece Unigram 250k, multilingual)
  mBERT      bert-multi   (WordPiece      120k, multilingual)
  tokenismo  v6           (Unigram LM    262k, our model)

Metrics:
  compression — chars per token (higher = better, fewer tokens = cheaper)
  throughput  — MB/s on 64 KB input (higher = better)
  RU/EN ratio — token count RU / token count EN (lower = better parity)

Usage:
  python scripts/benchmark_competitors.py [--no-hf]   # --no-hf skips XLM-R/mBERT
"""

from __future__ import annotations

import argparse
import io
import sys
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))

VOCAB_V6 = ROOT / "data" / "vocab" / "tokenismo_v6.vocab"
SAMPLES = {
    "EN":   (ROOT / "data" / "samples" / "sample_en.txt").read_text(encoding="utf-8"),
    "RU":   (ROOT / "data" / "samples" / "sample_ru.txt").read_text(encoding="utf-8"),
    "Code": (ROOT / "data" / "samples" / "sample_code.py").read_text(encoding="utf-8"),
}

# ── helpers ──────────────────────────────────────────────────────────────────

def make_64k(text: str) -> str:
    raw = text.encode("utf-8")
    return (raw * (65536 // len(raw) + 2))[:65536].decode("utf-8", "ignore")

def bench_throughput(fn, text: str, n: int = 15) -> float:
    fn(text[:512])          # warm up JIT / caches
    t0 = time.perf_counter()
    for _ in range(n):
        fn(text)
    return len(text.encode("utf-8")) * n / (time.perf_counter() - t0) / 1e6

def chars_per_token(fn, text: str) -> float:
    return len(text) / len(fn(text))

# ── tokenizer loaders ─────────────────────────────────────────────────────────

def load_tiktoken():
    import tiktoken
    return {
        "tiktoken cl100k": tiktoken.get_encoding("cl100k_base").encode,
        "tiktoken o200k":  tiktoken.get_encoding("o200k_base").encode,
    }

def load_hf_tokenizers():
    """Load XLM-R and mBERT via huggingface/tokenizers (no transformers needed)."""
    from tokenizers import Tokenizer
    result = {}
    for name, repo in [
        ("XLM-R (250k)",  "xlm-roberta-base"),
        ("mBERT (120k)",  "bert-base-multilingual-cased"),
    ]:
        try:
            tok = Tokenizer.from_pretrained(repo)
            # tokenizers returns Encoding objects; .ids is the token list
            result[name] = lambda text, t=tok: t.encode(text).ids
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  WARNING: could not load {name}: {e}")
    return result

def load_tokenismo():
    from tokenismo import TokeNismo
    tok = TokeNismo.from_file(str(VOCAB_V6))
    return {"tokenismo v6": tok.encode}

# ── main benchmark ────────────────────────────────────────────────────────────

def run(tokenizers: dict[str, callable]) -> dict:
    inputs_64k = {lang: make_64k(text) for lang, text in SAMPLES.items()}
    results = {}

    for name, fn in tokenizers.items():
        print(f"  Benchmarking {name} ...")
        cpt  = {lang: chars_per_token(fn, text) for lang, text in SAMPLES.items()}
        spd  = {lang: bench_throughput(fn, inputs_64k[lang]) for lang in SAMPLES}
        en_tokens = len(fn(SAMPLES["EN"]))
        ru_tokens = len(fn(SAMPLES["RU"]))
        ratio = ru_tokens / en_tokens if en_tokens > 0 else 0.0
        results[name] = {"cpt": cpt, "spd": spd, "ru_en_ratio": ratio}

    return results


def format_table(results: dict) -> str:
    LANGS = ["EN", "RU", "Code"]
    rows = []

    rows.append("### Compression (chars per token — higher is better)\n")
    rows.append(f"| Tokenizer          | Vocab  | {'EN':>6} | {'RU':>6} | {'Code':>6} | RU/EN ratio |")
    rows.append("|:-------------------|-------:|------:|------:|------:|------------:|")

    vocab_sizes = {
        "tiktoken cl100k":  "100k",
        "tiktoken o200k":   "200k",
        "XLM-R (250k)":     "250k",
        "mBERT (120k)":     "120k",
        "tokenismo v6":     "262k",
    }

    for name, r in results.items():
        c = r["cpt"]
        ratio = r["ru_en_ratio"]
        vs = vocab_sizes.get(name, "—")
        rows.append(
            f"| {name:18} | {vs:>6} | {c['EN']:>6.2f} | {c['RU']:>6.2f} | {c['Code']:>6.2f} | {ratio:>11.3f}x |"
        )

    rows.append("")
    rows.append("### Throughput (MB/s on 64 KB input — higher is better)\n")
    rows.append(f"| Tokenizer          | {'EN':>8} | {'RU':>8} | {'Code':>8} | vs o200k (EN) |")
    rows.append("|:-------------------|---------:|---------:|---------:|---------------:|")

    o200k_en = results.get("tiktoken o200k", {}).get("spd", {}).get("EN", 1.0)
    for name, r in results.items():
        s = r["spd"]
        ratio_spd = s["EN"] / o200k_en if o200k_en > 0 else 0
        rows.append(
            f"| {name:18} | {s['EN']:>8.1f} | {s['RU']:>8.1f} | {s['Code']:>8.1f} | {ratio_spd:>13.1f}x |"
        )

    rows.append("")
    rows.append("### tokenismo v6 vs tiktoken o200k (direct comparison)\n")
    v6   = results.get("tokenismo v6", {}).get("cpt", {})
    o200k = results.get("tiktoken o200k", {}).get("cpt", {})
    if v6 and o200k:
        for lang in LANGS:
            diff = v6[lang] - o200k[lang]
            pct  = diff / o200k[lang] * 100
            mark = "✅ better" if diff >= 0 else "❌ worse"
            rows.append(f"- **{lang}**: {v6[lang]:.2f} vs {o200k[lang]:.2f} cpt  ({pct:+.1f}%)  {mark}")

    return "\n".join(rows)


def print_table(results: dict):
    LANGS = ["EN", "RU", "Code"]
    print(f"\n{'Tokenizer':20}  {'EN':>6}  {'RU':>6}  {'Code':>6}  {'RU/EN':>8}  │  {'EN spd':>8}  {'RU spd':>8}  {'Code spd':>9}")
    print("─" * 90)
    for name, r in results.items():
        c, s = r["cpt"], r["spd"]
        print(
            f"{name:20}  {c['EN']:6.2f}  {c['RU']:6.2f}  {c['Code']:6.2f}  "
            f"{r['ru_en_ratio']:8.3f}x  │  {s['EN']:8.1f}  {s['RU']:8.1f}  {s['Code']:9.1f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-hf", action="store_true", help="Skip XLM-R and mBERT (no HF download)")
    args = p.parse_args()

    print("Loading tokenizers...")
    tokenizers: dict[str, callable] = {}
    tokenizers.update(load_tiktoken())
    if not args.no_hf:
        tokenizers.update(load_hf_tokenizers())
    tokenizers.update(load_tokenismo())

    print(f"\nRunning benchmarks ({len(SAMPLES)} languages x {len(tokenizers)} tokenizers)...")
    results = run(tokenizers)

    print_table(results)

    md = format_table(results)
    out = ROOT / "scripts" / "benchmark_results.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nMarkdown results saved → {out.relative_to(ROOT)}")
    print("\nPaste into README.md Benchmarks section.")
    return results


if __name__ == "__main__":
    main()
