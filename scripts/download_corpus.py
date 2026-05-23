"""
Download corpus subsets from HuggingFace for 262k vocab training.

Downloads Parquet shards directly (more reliable than streaming):
  - Wikipedia EN  20231101.en  : up to --en-docs articles
  - Wikipedia RU  20231101.ru  : up to --ru-docs articles
  - The Stack Smol per language: up to --code-docs files each

Output: data/corpus/{wiki_en,wiki_ru,stack_python,stack_go,stack_rust,stack_cpp,stack_ts}.jsonl.gz
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

# Language name → (the-stack-smol data_dir, output filename)
STACK_LANGS = {
    "python":     ("data/python",     "stack_python.jsonl.gz"),
    "go":         ("data/go",         "stack_go.jsonl.gz"),
    "rust":       ("data/rust",       "stack_rust.jsonl.gz"),
    "cpp":        ("data/c++",        "stack_cpp.jsonl.gz"),
    "typescript": ("data/typescript", "stack_ts.jsonl.gz"),
}


def download_wikipedia(language: str, max_docs: int, out_path: Path) -> int:
    from datasets import load_dataset

    print(f"\n[Wikipedia/{language}] downloading up to {max_docs:,} docs -> {out_path.name}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

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


def download_stack_lang(lang: str, data_dir: str, max_docs: int, out_path: Path, hf_token: str | None) -> int:
    import os
    from datasets import load_dataset

    print(f"\n[the-stack-smol/{lang}] downloading up to {max_docs:,} files -> {out_path.name}")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Set env var so HF library uses it for all internal requests.
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    kwargs: dict = dict(streaming=True, split="train")
    if hf_token:
        kwargs["token"] = hf_token

    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir=data_dir, **kwargs)
        return _write_code_docs(ds, "content", max_docs, out_path)
    except Exception as e:
        print(f"  WARNING: the-stack-smol failed ({e})")
        print(f"  Hint: accept the BigCode SAS license at https://huggingface.co/datasets/bigcode/the-stack-smol")
        print(f"  Falling back to code-search-net/code_search_net ...")
        return _download_codesearch_net(lang, max_docs, out_path)


def _download_codesearch_net(lang: str, max_docs: int, out_path: Path) -> int:
    """Fallback: code-search-net/code_search_net — ungated, GitHub functions.
    Languages: python, go, java, javascript, ruby, php.
    For rust/cpp/typescript we substitute the closest available language.
    """
    from datasets import load_dataset

    # Map our lang names to code_search_net subset names.
    csn_map = {
        "python": "python", "go": "go",
        "rust": "go",           # closest compiled systems lang available
        "cpp": "java",          # closest statically-typed lang available
        "typescript": "javascript",
    }
    csn_lang = csn_map.get(lang)
    if csn_lang is None:
        print(f"  No code-search-net fallback for {lang}, skipping.")
        return 0

    substitute = (csn_lang != lang)
    if substitute:
        print(f"  No {lang} in code-search-net; using {csn_lang} as substitute.")

    ds = load_dataset("code-search-net/code_search_net", csn_lang,
                      split="train", streaming=True)

    written = 0
    skipped = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for doc in ds:
            if written >= max_docs:
                break
            content = doc.get("whole_func_string", "").strip()
            if len(content) < 80:
                skipped += 1
                continue
            f.write(json.dumps({"content": content}, ensure_ascii=False) + "\n")
            written += 1
            if written % 5_000 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(f"  {written:,}/{max_docs:,}  {rate:.0f} files/s")

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Done (code-search-net/{csn_lang}): {written:,} files, {size_mb:.1f} MB ({elapsed:.0f}s)")
    return written


def _write_code_docs(ds, field: str, max_docs: int, out_path: Path) -> int:
    written = 0
    skipped = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for doc in ds:
            if written >= max_docs:
                break
            content = doc.get(field, "").strip()
            if len(content) < 80:
                skipped += 1
                continue
            f.write(json.dumps({"content": content}, ensure_ascii=False) + "\n")
            written += 1
            if written % 5_000 == 0:
                elapsed = time.time() - t0
                size_mb = out_path.stat().st_size / 1e6
                rate = written / elapsed if elapsed > 0 else 0
                eta = (max_docs - written) / rate if rate > 0 else 0
                print(f"  {written:,}/{max_docs:,} files  {size_mb:.1f}MB  {rate:.0f} files/s  ETA {eta:.0f}s")
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Done: {written:,} files ({skipped:,} skipped), {size_mb:.1f} MB compressed ({elapsed:.0f}s)")
    return written


def main():
    p = argparse.ArgumentParser(description="Download corpus for TokeNismo Mix v6 training")
    p.add_argument("--en-docs",   type=int, default=100_000)
    p.add_argument("--ru-docs",   type=int, default=50_000)
    p.add_argument("--code-docs", type=int, default=40_000,
                   help="Max files per code language (5 languages = 200k total)")
    p.add_argument("--langs",     nargs="+", default=list(STACK_LANGS.keys()),
                   choices=list(STACK_LANGS.keys()),
                   help="Code languages to download")
    p.add_argument("--hf-token",  default=None,
                   help="HuggingFace token for gated datasets (bigcode/the-stack-smol)")
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
        for lang in args.langs:
            data_dir, out_name = STACK_LANGS[lang]
            out_path = CORPUS_DIR / out_name
            if out_path.exists():
                size_mb = out_path.stat().st_size / 1e6
                print(f"\n[{lang}] {out_name} already exists ({size_mb:.0f} MB) — skipping. "
                      f"Delete to re-download.")
                continue
            download_stack_lang(lang, data_dir, args.code_docs, out_path, args.hf_token)

    elapsed = time.time() - total_t0
    print(f"\nAll downloads done in {elapsed/60:.1f} min")
    print("Next step:")
    print("  cargo run --release --bin tokenismo-train -- "
          "--config configs/corpus_v6.yaml "
          "--vocab-size 262144 --min-freq 5 --max-docs 150000 "
          "--output data/vocab/tokenismo_v6.vocab")


if __name__ == "__main__":
    main()
