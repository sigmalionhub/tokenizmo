"""
TokeNismo — multilingual tokenizer with Unigram LM + Viterbi DP segmentation.

The ``TokeNismo`` class wraps the Rust encoder/decoder and exposes a simple
Python API for single-text and batch encoding.

Example::

    from tokenismo import TokeNismo

    tok = TokeNismo.from_file("data/vocab/tokenismo_small.vocab")
    ids = tok.encode("Hello мир")          # list[int]
    text = tok.decode(ids)                  # "Hello мир"

    batch_ids = tok.encode_batch(["Hello", "мир"])   # list[list[int]]
    texts = tok.decode_batch(batch_ids)              # list[str]

Token IDs may carry a ``LEADING_SPACE_FLAG`` (bit 22) indicating the token
was preceded by a space.  The decoder handles this transparently.
"""

try:
    from .tokenismo import TokeNismo
except ImportError:
    raise ImportError(
        "tokenismo native extension not built. "
        "Run: cd tokenismo-py && maturin develop --release"
    )

__all__ = ["TokeNismo"]
__version__ = "0.1.0"
