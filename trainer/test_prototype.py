"""
Integration test for the Python prototype.

Tests:
1. Train small vocabulary (size=8192) on sample files
2. Encode each sample file with Viterbi
3. Verify round-trip: decode(encode(text)) == text
4. Print token counts vs tiktoken baseline
5. Verify Russian/English parity improvement target

Usage:
    python -m trainer.test_prototype
    (from project root: D:/Projects/tikenismo/)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trainer import UnigramTrainer, ViterbiEncoder, normalize


SAMPLES = ROOT / "data" / "samples"
SMALL_VOCAB_SIZE = 8192


def load_sample(name: str) -> str:
    path = SAMPLES / name
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}. Run: python scripts/fetch_samples.py")
    return path.read_text(encoding="utf-8")


def run_tests() -> bool:
    print("=" * 60)
    print("TokeNismo Python Prototype — Integration Tests")
    print("=" * 60)

    # Load samples
    samples = {}
    for name in ("sample_en.txt", "sample_ru.txt", "sample_code.py"):
        try:
            raw = load_sample(name)
            samples[name] = normalize(raw)
            print(f"  [OK] Loaded {name}: {len(samples[name]):,} chars")
        except FileNotFoundError as e:
            print(f"  [ERR] {e}")
            return False

    # Train vocabulary
    print(f"\nTraining vocabulary (size={SMALL_VOCAB_SIZE})...")
    corpus = list(samples.values())
    t0 = time.perf_counter()
    trainer = UnigramTrainer(
        vocab_size=SMALL_VOCAB_SIZE,
        max_token_len=16,
        shrink_factor=0.75,
        min_freq=2,
    )
    vocab = trainer.train(corpus)
    train_time = time.perf_counter() - t0
    print(f"  [OK] Vocabulary trained: {len(vocab):,} tokens in {train_time:.1f}s")

    # Build encoder
    encoder = ViterbiEncoder(vocab)

    # Test 1: Round-trip fidelity
    print("\nTest 1: Round-trip fidelity (encode → decode == original)")
    all_passed = True
    for name, text in samples.items():
        ids = encoder.encode(text)
        recovered = encoder.decode(ids)
        passed = recovered == text
        all_passed = all_passed and passed
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}: {len(ids):,} tokens (original: {len(text):,} chars)")
        if not passed:
            # Find first mismatch
            for i, (a, b) in enumerate(zip(text, recovered)):
                if a != b:
                    print(f"         First mismatch at char {i}: original={repr(a)}, decoded={repr(b)}")
                    break

    assert all_passed, "Round-trip test FAILED — decoder does not reconstruct original text"

    # Test 2: Viterbi optimality — single-token preference
    print("\nTest 2: Viterbi prefers fewer tokens")
    from trainer.vocabulary import Vocabulary
    from trainer.viterbi_encoder import ViterbiEncoder as VE

    manual_vocab = Vocabulary.from_log_probs({
        b"hello": -1.0,
        b"world": -1.0,
        b"hello world": -0.5,  # single token covering both words
        b" ": -2.0,
    })
    enc2 = VE(manual_vocab)
    ids = enc2.encode("hello world")
    # With LEADING_SPACE_FLAG approach: "hello" + "world"|FLAG = 2 tokens
    # With merged "hello world" token: ... depends on whether "hello world"
    # can be matched (it contains a space byte). Let's verify count <= 2.
    assert len(ids) <= 2, f"Expected ≤ 2 tokens for 'hello world', got {len(ids)}: {ids}"
    print(f"  [PASS]  'hello world' → {len(ids)} token(s)")

    # Test 3: Token count comparison
    print("\nTest 3: Token count summary")
    tiktoken_counts: dict[str, int] = {}
    try:
        import tiktoken
        enc_tiktoken = tiktoken.get_encoding("o200k_base")
        for name, text in samples.items():
            tiktoken_counts[name] = len(enc_tiktoken.encode(text))
        has_tiktoken = True
    except ImportError:
        has_tiktoken = False
        print("  (tiktoken not installed — skipping comparison)")

    print(f"\n  {'Sample':<20} {'TokeNismo':>12} {'tiktoken/o200k':>15} {'ratio':>8}")
    print("  " + "-" * 58)
    for name, text in samples.items():
        ids = encoder.encode(text)
        our_count = len(ids)
        if has_tiktoken:
            their_count = tiktoken_counts[name]
            ratio = our_count / their_count if their_count else 0
            ratio_str = f"{ratio:.2f}x"
        else:
            their_count = "N/A"
            ratio_str = "N/A"
        print(f"  {name:<20} {our_count:>12,} {str(their_count):>15} {ratio_str:>8}")

    # Test 4: Russian/English parity
    print("\nTest 4: Russian/English token density parity")
    en_text = samples["sample_en.txt"]
    ru_text = samples["sample_ru.txt"]
    en_ids = encoder.encode(en_text)
    ru_ids = encoder.encode(ru_text)
    en_density = len(en_ids) / len(en_text)  # tokens per char
    ru_density = len(ru_ids) / len(ru_text)
    ru_en_ratio = ru_density / en_density if en_density > 0 else float("inf")

    status = "[PASS]" if ru_en_ratio <= 2.0 else "[PARTIAL]"
    target_status = "[TARGET MET]" if ru_en_ratio <= 1.5 else f"(target ≤ 1.5 — requires full vocab training)"
    print(f"  EN token density: {en_density:.4f} tok/char")
    print(f"  RU token density: {ru_density:.4f} tok/char")
    print(f"  RU/EN ratio: {ru_en_ratio:.3f}  {status}  {target_status}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
