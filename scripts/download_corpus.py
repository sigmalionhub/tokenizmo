"""
Download corpus subsets from HuggingFace for 262k vocab training.

Downloads Parquet shards directly (more reliable than streaming):
  - Wikipedia EN  20231101.en  : up to --en-docs articles
  - Wikipedia RU  20231101.ru  : up to --ru-docs articles
  - The Stack dedup Python     : up to --code-docs files

Output: data/corpus/{wiki_en,wiki_ru,stack_python}.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"


def download_wikipedia(language: str, max_docs: int, out_path: Path) -> int:
    from datasets import load_dataset

    print(f"\n[Wikipedia/{language}] downloading up to {max_docs:,} docs -> {out_path.name}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Use slice notation — downloads only the needed Parquet shards.
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{language}",
        split=f"train[:{max_docs}]",
    )

    written = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "").strip()
            if len(text) < 200:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  {written:,} docs written...", end="\r")

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Done: {written:,} docs, {size_mb:.1f} MB compressed ({elapsed:.0f}s)")
    return written


def download_the_stack(max_docs: int, out_path: Path) -> int:
    from datasets import load_dataset

    print(f"\n[The Stack/Python] downloading up to {max_docs:,} files -> {out_path.name}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split=f"train[:{max_docs}]",
    )

    written = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for doc in ds:
            content = doc.get("content", "").strip()
            if len(content) < 100:
                continue
            f.write(json.dumps({"content": content}, ensure_ascii=False) + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  {written:,} files written...", end="\r")

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Done: {written:,} files, {size_mb:.1f} MB compressed ({elapsed:.0f}s)")
    return written


def main():
    p = argparse.ArgumentParser(description="Download corpus for TokeNismo training")
    p.add_argument("--en-docs",   type=int, default=100_000)
    p.add_argument("--ru-docs",   type=int, default=100_000)
    p.add_argument("--code-docs", type=int, default=50_000)
    p.add_argument("--skip-en",   action="store_true")
    p.add_argument("--skip-ru",   action="store_true")
    p.add_argument("--skip-code", action="store_true")
    args = p.parse_args()

    total_t0 = time.time()

    if not args.skip_en:
        download_wikipedia("en", args.en_docs, CORPUS_DIR / "wiki_en.jsonl.gz")

    if not args.skip_ru:
        download_wikipedia("ru", args.ru_docs, CORPUS_DIR / "wiki_ru.jsonl.gz")

    if not args.skip_code:
        download_the_stack(args.code_docs, CORPUS_DIR / "stack_python.jsonl.gz")

    elapsed = time.time() - total_t0
    print(f"\nAll downloads done in {elapsed/60:.1f} min")
    print("Next step:")
    print("  python trainer/train.py --corpus-config configs/corpus_full.yaml "
          "--vocab-size 262144 --output data/vocab/tokenismo.vocab")


if __name__ == "__main__":
    main()
