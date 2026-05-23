import sys, time
sys.path.insert(0, "python")
from tokenismo import TokeNismo
import tiktoken

SAMPLES = {
    "EN":   open("data/samples/sample_en.txt",  encoding="utf-8").read(),
    "RU":   open("data/samples/sample_ru.txt",  encoding="utf-8").read(),
    "Code": open("data/samples/sample_code.py", encoding="utf-8").read(),
}

VOCABS = [
    ("tokenismo v4",      "data/vocab/tokenismo_262k_rust_v4.vocab"),
    ("tokenismo v5+code", "data/vocab/tokenismo_v5_code30.vocab"),
]

tik = {
    "tiktoken cl100k": tiktoken.get_encoding("cl100k_base"),
    "tiktoken o200k":  tiktoken.get_encoding("o200k_base"),
}

print("\n=== COMPRESSION (chars per token — higher is better) ===")
print(f"{'':22}  {'EN':>7}  {'RU':>7}  {'Code':>7}  {'RU/EN':>7}")
print("-" * 56)

results = {}
for name, enc in tik.items():
    row = {lang: len(text)/len(enc.encode(text)) for lang, text in SAMPLES.items()}
    ru_en = (len(enc.encode(SAMPLES["RU"]))/len(SAMPLES["RU"])) / \
            (len(enc.encode(SAMPLES["EN"]))/len(SAMPLES["EN"]))
    results[name] = row
    print(f"{name:22}  {row['EN']:7.2f}  {row['RU']:7.2f}  {row['Code']:7.2f}  {ru_en:7.3f}x")

for name, path in VOCABS:
    tok = TokeNismo.from_file(path)
    row = {lang: len(text)/len(tok.encode(text)) for lang, text in SAMPLES.items()}
    ru_en = (len(tok.encode(SAMPLES["RU"]))/len(SAMPLES["RU"])) / \
            (len(tok.encode(SAMPLES["EN"]))/len(SAMPLES["EN"]))
    results[name] = row
    print(f"{name:22}  {row['EN']:7.2f}  {row['RU']:7.2f}  {row['Code']:7.2f}  {ru_en:7.3f}x")

print("\n=== THROUGHPUT via Python API (MB/s, 64 KB input, warm cache) ===")
print(f"{'':22}  {'EN':>9}  {'RU':>9}  {'Code':>9}")
print("-" * 54)

def make_input(text, target=65536):
    raw = text.encode("utf-8")
    repeated = (raw * (target // len(raw) + 2))[:target]
    return repeated.decode("utf-8", "ignore")

inputs = {lang: make_input(text) for lang, text in SAMPLES.items()}

def bench(fn, text, n=30):
    fn(text[:500])  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn(text)
    return len(text.encode("utf-8")) * n / (time.perf_counter() - t0) / 1e6

tik_speeds = {}
for name, enc in tik.items():
    row = {lang: bench(enc.encode, inputs[lang]) for lang in SAMPLES}
    tik_speeds[name] = row
    print(f"{name:22}  {row['EN']:9.1f}  {row['RU']:9.1f}  {row['Code']:9.1f}")

tok_speeds = {}
for name, path in VOCABS:
    tok = TokeNismo.from_file(path)
    row = {lang: bench(tok.encode, inputs[lang]) for lang in SAMPLES}
    tok_speeds[name] = row
    print(f"{name:22}  {row['EN']:9.1f}  {row['RU']:9.1f}  {row['Code']:9.1f}")

print("\n=== TOKENISMO v5 vs tiktoken o200k ===")
o200k = results["tiktoken o200k"]
v5    = results["tokenismo v5+code"]
o200k_sp = tik_speeds["tiktoken o200k"]
v5_sp    = tok_speeds["tokenismo v5+code"]
for lang in ["EN", "RU", "Code"]:
    cpt_diff = v5[lang] - o200k[lang]
    cpt_pct  = cpt_diff / o200k[lang] * 100
    spd_x    = v5_sp[lang] / o200k_sp[lang]
    cpt_sign = "+" if cpt_diff > 0 else ""
    cpt_mark = "✓ BETTER" if cpt_diff > 0 else "✗ WORSE"
    spd_mark = "✓ FASTER" if spd_x > 1 else "✗ SLOWER"
    print(f"  {lang:4}  compression: {cpt_sign}{cpt_pct:+.1f}% {cpt_mark}   "
          f"speed: {spd_x:.1f}x {spd_mark}")
