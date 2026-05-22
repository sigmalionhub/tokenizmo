"""
Fetch sample data files for benchmarking.
Downloads ~50KB each of English text, Russian text, and Python code.

Usage:
    python scripts/fetch_samples.py
"""

import urllib.request
import urllib.error
import re
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)


def _english_fallback() -> str:
    return (
        "Natural language processing (NLP) is an interdisciplinary subfield of linguistics, "
        "computer science, and artificial intelligence concerned with the interactions between "
        "computers and human language, in particular how to program computers to process and "
        "analyze large amounts of natural language data. The goal is a computer capable of "
        "understanding the contents of documents, including the contextual nuances of the "
        "language within them.\n\n"
        "Tokenization is the process of breaking a stream of text up into words, phrases, "
        "symbols, or other meaningful elements called tokens. The list of tokens becomes input "
        "for further processing such as parsing or text mining. Tokenization is useful both in "
        "linguistics (where it is a form of text segmentation), and in computer science, where "
        "it forms part of lexical analysis.\n\n"
        "Byte pair encoding (BPE) or digram coding is a simple form of data compression in "
        "which the most common pair of consecutive bytes of data is replaced with a byte that "
        "does not occur within that data. The use of BPE for subword tokenization in natural "
        "language processing was introduced in 2015 and has since become the dominant approach "
        "for training tokenizers for large language models.\n\n"
        "The Viterbi algorithm is a dynamic programming algorithm for obtaining the maximum a "
        "posteriori probability estimate of the most likely sequence of hidden states. It has "
        "found universal application in decoding convolutional codes used in digital cellular "
        "networks, and is now commonly used in speech recognition and computational linguistics.\n\n"
    ) * 8


def _russian_fallback() -> str:
    return (
        "Токенизация (разбиение на токены) — задача разбиения текста на значимые единицы. "
        "Токеном могут быть слово, число или знак препинания. В компьютерной лингвистике "
        "задача токенизации является частью более широкой задачи лексического анализа текста.\n\n"
        "Обработка естественного языка (ОЕЯ, Natural Language Processing, NLP) — общее "
        "направление искусственного интеллекта и математической лингвистики, изучает проблемы "
        "компьютерного анализа и синтеза естественного языка. Применительно к искусственному "
        "интеллекту анализ означает понимание языка, синтез — генерацию грамотного текста.\n\n"
        "Русский язык — язык восточнославянской группы славянской ветви индоевропейской языковой "
        "семьи, национальный язык русского народа. Является одним из наиболее распространённых "
        "языков мира — шестым среди всех языков мира по общей численности говорящих.\n\n"
        "Морфология — раздел лингвистики, изучающий слово, его строение и формы. Морфология "
        "рассматривает такие понятия, как часть речи, категории рода, числа, падежа, времени, "
        "вида, залога, лица. В русском языке богатая система флексий позволяет выражать "
        "грамматические значения непосредственно в структуре слова.\n\n"
        "Машинное обучение — класс методов искусственного интеллекта, характерной чертой "
        "которых является не прямое решение задачи, а обучение в процессе применения решений "
        "множества сходных задач. Нейронные сети — вычислительные системы, вдохновлённые "
        "биологическими нейронными сетями мозга животных.\n\n"
    ) * 8


def _code_fallback() -> str:
    return '''\
#!/usr/bin/env python3
"""Sample Python code for tokenizer benchmarking."""

from typing import Iterator, Optional
import dataclasses
import math


@dataclasses.dataclass
class Token:
    text: str
    log_prob: float
    token_id: int


class UnigramVocabulary:
    """Unigram language model vocabulary."""

    def __init__(self, tokens: dict[str, float]) -> None:
        self._tokens = tokens
        self._sorted = sorted(tokens.items(), key=lambda x: -x[1])

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    def log_prob(self, token: str) -> float:
        return self._tokens.get(token, float("-inf"))

    def prune(self, keep_fraction: float) -> "UnigramVocabulary":
        n = max(1, int(len(self._tokens) * keep_fraction))
        return UnigramVocabulary(dict(self._sorted[:n]))


def viterbi_encode(text: str, vocab: UnigramVocabulary) -> list[int]:
    """Encode text using Viterbi DP for minimum token count."""
    n = len(text)
    INF = float("inf")
    # dp[i] = (token_count, neg_logprob, prev_pos, token_hash)
    dp = [(INF, INF, -1, -1)] * (n + 1)
    dp[0] = (0.0, 0.0, -1, -1)

    for i in range(n):
        if dp[i][0] == INF:
            continue
        for length in range(1, min(32, n - i + 1)):
            substr = text[i : i + length]
            if substr in vocab:
                lp = vocab.log_prob(substr)
                new_count = dp[i][0] + 1
                new_lp = dp[i][1] - lp
                if (new_count, new_lp) < (dp[i + length][0], dp[i + length][1]):
                    dp[i + length] = (new_count, new_lp, i, hash(substr))

    tokens = []
    pos = n
    while pos > 0:
        _, _, prev, tid = dp[pos]
        tokens.append(tid)
        pos = prev
    return list(reversed(tokens))


def estimate_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log(c / total)
        for c in counts.values()
        if c > 0
    )


def sliding_window(text: str, max_len: int = 16) -> Iterator[str]:
    n = len(text)
    for i in range(n):
        for length in range(1, min(max_len + 1, n - i + 1)):
            yield text[i : i + length]


class TrieNode:
    __slots__ = ("children", "token_id", "log_prob")

    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.token_id: Optional[int] = None
        self.log_prob: float = float("-inf")


class VocabTrie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    def insert(self, token: str, log_prob: float) -> int:
        node = self.root
        for byte in token.encode("utf-8"):
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
        node.token_id = self._size
        node.log_prob = log_prob
        self._size += 1
        return node.token_id

    def __len__(self) -> int:
        return self._size
''' * 2


SAMPLES_TO_FETCH = [
    (
        "sample_en.txt",
        "https://en.wikipedia.org/wiki/Special:Export/Tokenization_(lexical_analysis)",
        _english_fallback,
    ),
    (
        "sample_ru.txt",
        "https://ru.wikipedia.org/wiki/Special:Export/%D0%A2%D0%BE%D0%BA%D0%B5%D0%BD%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F",
        _russian_fallback,
    ),
    (
        "sample_code.py",
        None,
        _code_fallback,
    ),
]


def fetch_or_fallback(name: str, url: Optional[str], fallback_fn) -> None:
    path = SAMPLES / name
    if path.exists():
        print(f"  ✓ {name} already exists ({path.stat().st_size // 1024}KB)")
        return

    text = None
    if url:
        try:
            print(f"  Fetching {name}...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            text = text[:60_000]
        except Exception as exc:
            print(f"  ✗ Fetch failed ({exc}), using built-in fallback")

    if not text:
        text = fallback_fn()

    path.write_text(text, encoding="utf-8")
    print(f"  ✓ {name} written ({len(text) // 1024}KB)")


def main() -> None:
    print("Fetching sample data files...")
    for name, url, fallback_fn in SAMPLES_TO_FETCH:
        fetch_or_fallback(name, url, fallback_fn)
    print(f"\nSamples ready in {SAMPLES}")


if __name__ == "__main__":
    main()
