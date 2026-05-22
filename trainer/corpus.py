"""
Streaming corpus pipeline for TokeNismo vocabulary training.

Supports mixed EN/RU/Code sources with configurable blend ratios.
Streams documents without loading the entire corpus into RAM.
"""

from __future__ import annotations

import gzip
import bz2
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Callable


@dataclass
class CorpusSource:
    path: Path
    language: str           # "en", "ru", "code"
    weight: float = 1.0     # sampling weight
    reader: str = "text"    # "text" | "jsonl" | "jsonl_gz" | "text_gz"
    text_field: str = "text"  # for jsonl: which field contains the text


class CorpusStream:
    """
    Streams documents from multiple sources with configurable mix ratios.
    Uses reservoir / round-robin sampling to honor weights without loading
    everything into memory.

    Usage:
        stream = CorpusStream(sources, mix_ratios={"en": 0.4, "ru": 0.4, "code": 0.2})
        for doc in stream:
            process(doc)
    """

    def __init__(
        self,
        sources: list[CorpusSource],
        mix_ratios: dict[str, float] | None = None,
        chunk_size_mb: int = 64,
        seed: int = 42,
    ) -> None:
        self.sources = sources
        self.chunk_size_mb = chunk_size_mb
        self._rng = random.Random(seed)

        # Normalize mix_ratios into per-source weights
        if mix_ratios:
            for src in self.sources:
                src.weight = mix_ratios.get(src.language, src.weight)

        total_w = sum(s.weight for s in self.sources)
        self._probs = [s.weight / total_w for s in self.sources]

    def __iter__(self) -> Iterator[str]:
        # Open iterators for each source
        iters = [_open_source(src) for src in self.sources]
        exhausted = [False] * len(self.sources)

        while not all(exhausted):
            # Weighted random pick of a non-exhausted source
            active = [(i, p) for i, p in enumerate(self._probs) if not exhausted[i]]
            if not active:
                break
            idxs, probs = zip(*active)
            total = sum(probs)
            r = self._rng.random() * total
            chosen = idxs[0]
            acc = 0.0
            for idx, p in zip(idxs, probs):
                acc += p
                if r <= acc:
                    chosen = idx
                    break

            try:
                doc = next(iters[chosen])
                if doc and doc.strip():
                    yield doc
            except StopIteration:
                exhausted[chosen] = True

    def estimate_size(self) -> int:
        """Estimate total corpus size in bytes (sum of file sizes)."""
        total = 0
        for src in self.sources:
            try:
                total += src.path.stat().st_size
            except OSError:
                pass
        return total


def _open_source(src: CorpusSource) -> Iterator[str]:
    """Open a corpus source and yield one document (string) at a time."""
    path = src.path
    if not path.exists():
        return

    if src.reader == "text":
        yield from _read_text(path)
    elif src.reader == "text_gz":
        yield from _read_text_gz(path)
    elif src.reader == "jsonl":
        yield from _read_jsonl(path, src.text_field)
    elif src.reader == "jsonl_gz":
        yield from _read_jsonl_gz(path, src.text_field)
    else:
        raise ValueError(f"Unknown reader: {src.reader}")


def _read_text(path: Path) -> Iterator[str]:
    """Yield paragraphs (blank-line separated) from a plain text file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        buf: list[str] = []
        for line in f:
            if line.strip():
                buf.append(line.rstrip("\n"))
            elif buf:
                yield "\n".join(buf)
                buf = []
        if buf:
            yield "\n".join(buf)


def _read_text_gz(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        buf: list[str] = []
        for line in f:
            if line.strip():
                buf.append(line.rstrip("\n"))
            elif buf:
                yield "\n".join(buf)
                buf = []
        if buf:
            yield "\n".join(buf)


def _read_jsonl(path: Path, text_field: str) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text:
                    yield str(text)
            except json.JSONDecodeError:
                pass


def _read_jsonl_gz(path: Path, text_field: str) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text:
                    yield str(text)
            except json.JSONDecodeError:
                pass


def make_sample_corpus() -> CorpusStream:
    """
    Create a CorpusStream from the bundled sample files.
    Used for testing and small-scale training runs.
    """
    root = Path(__file__).parent.parent
    samples = root / "data" / "samples"
    sources = [
        CorpusSource(samples / "sample_en.txt", language="en", weight=0.4),
        CorpusSource(samples / "sample_ru.txt", language="ru", weight=0.4),
        CorpusSource(samples / "sample_code.py", language="code", weight=0.2),
    ]
    return CorpusStream(sources, seed=42)
