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

for i, text in enumerate(cases):
    try:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        rt = "OK" if decoded == text else "ROUNDTRIP_FAIL"
        print(f"[{i:02d}] {rt}  tokens={len(ids)}  input={repr(text[:50])}")
    except Exception as e:
        print(f"[{i:02d}] EXCEPTION: {e}")
        print(f"       input={repr(text[:80])}")
        b = text.encode("utf-8")
        print(f"       bytes ({len(b)}): {b[:40].hex()}")
        break
