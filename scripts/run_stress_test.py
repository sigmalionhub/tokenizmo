"""Run all test_cases.json through a tokenismo vocab and verify round-trips."""
import argparse, json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))
from tokenismo import TokeNismo

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", default=str(ROOT / "data" / "vocab" / "tokenismo_v6.vocab"))
parser.add_argument("--cases", default=str(ROOT / "data" / "test_cases.json"))
args = parser.parse_args()

with open(args.cases, encoding="utf-8") as f:
    cases = json.load(f)

tok = TokeNismo.from_file(args.vocab)

ok = 0
failures = []
for i, text in enumerate(cases):
    try:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        if decoded == text:
            ok += 1
        else:
            failures.append((i, "ROUNDTRIP_FAIL", repr(text[:60]), repr(decoded[:60])))
    except Exception as e:
        failures.append((i, f"EXCEPTION: {e}", repr(text[:60]), ""))

print(f"Total: {len(cases)}  OK: {ok}  FAIL: {len(failures)}")
if failures:
    # Separate OOV (expected) from real bugs.
    oov, bugs = [], []
    for item in failures:
        i, reason, text, got = item
        if reason == "ROUNDTRIP_FAIL" and "<unk>" in got:
            oov.append(item)
        else:
            bugs.append(item)

    if oov:
        print(f"\nOOV (expected — characters outside vocabulary, <unk> emitted):")
        for i, reason, text, got in oov:
            print(f"  [{i:02d}] {text[:60]}")

    if bugs:
        print(f"\nBUGS ({len(bugs)}):")
        for i, reason, text, got in bugs:
            print(f"  [{i:02d}] {reason}")
            print(f"        input:   {text[:80]}")
            if got:
                print(f"        decoded: {got[:80]}")
    else:
        print("No real bugs — all failures are expected OOV round-trip mismatches.")
else:
    print("ALL PASSED — encode/decode round-trip OK on all stress cases")
