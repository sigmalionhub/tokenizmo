import sys, time
sys.path.insert(0, "python")
from tokenismo import TokeNismo
import tiktoken

SAMPLES = {
    "EN":   open("data/samples/sample_en.txt",  encoding="utf-8").read(),
    "RU":   open("data/samples/sample_ru.txt",  encoding="utf-8").read(),
    "Code": open("data/samples/sample_code.py", encoding="utf-8").read(),
}

def make_input(text, target=65536):
    raw = text.encode("utf-8")
    return (raw * (target // len(raw) + 2))[:target].decode("utf-8", "ignore")

inputs = {lang: make_input(text) for lang, text in SAMPLES.items()}

def bench(fn, text, n=10):
    fn(text[:500])
    t0 = time.perf_counter()
    for _ in range(n):
        fn(text)
    return len(text.encode("utf-8")) * n / (time.perf_counter() - t0) / 1e6

o200k  = tiktoken.get_encoding("o200k_base")
tok_v5 = TokeNismo.from_file("data/vocab/tokenismo_v5_code30.vocab")
tok_v6 = TokeNismo.from_file("data/vocab/tokenismo_v6.vocab")

MODELS = [
    ("o200k",   o200k.encode),
    ("v5+code", tok_v5.encode),
    ("v6",      tok_v6.encode),
]

print("=== COMPRESSION (chars/token, higher=better) ===")
print(f"{'':10}  {'EN':>6}  {'RU':>6}  {'Code':>6}")
cpt = {}
for name, fn in MODELS:
    row = {lang: len(text) / len(fn(text)) for lang, text in SAMPLES.items()}
    cpt[name] = row
    print(f"{name:10}  {row['EN']:6.2f}  {row['RU']:6.2f}  {row['Code']:6.2f}")

print()
print("=== THROUGHPUT MB/s (64KB, 10 iters) ===")
print(f"{'':10}  {'EN':>6}  {'RU':>6}  {'Code':>6}")
for name, fn in MODELS:
    row = {lang: bench(fn, inputs[lang]) for lang in SAMPLES}
    print(f"{name:10}  {row['EN']:6.1f}  {row['RU']:6.1f}  {row['Code']:6.1f}")

print()
print("=== v6 vs o200k ===")
for lang in ["EN", "RU", "Code"]:
    diff = cpt["v6"][lang] - cpt["o200k"][lang]
    pct  = diff / cpt["o200k"][lang] * 100
    sign = "+" if diff >= 0 else ""
    mark = "BETTER" if diff >= 0 else "WORSE"
    print(f"  {lang:4}  {sign}{pct:+.1f}%  {mark}")
