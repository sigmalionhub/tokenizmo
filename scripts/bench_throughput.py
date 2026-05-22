"""Quick throughput benchmark for TokeNismo Rust encoder."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))

from tokenismo import TokeNismo

tok = TokeNismo.from_file(str(ROOT / "data/vocab/tokenismo_small.vocab"))
data = (ROOT / "data/samples/sample_en.txt").read_text(encoding="utf-8") * 100
data_bytes = len(data.encode("utf-8"))

tok.encode(data[:1000])  # warmup

times = []
for _ in range(5):
    t0 = time.perf_counter()
    ids = tok.encode(data)
    times.append(time.perf_counter() - t0)

best = min(times)
mb_s = data_bytes / best / 1e6
print(f"Throughput (best of 5): {mb_s:.0f} MB/s")
print(f"Input: {data_bytes/1024:.0f} KB -> {len(ids):,} tokens")
print(f"All runs (ms): {[f'{t*1000:.0f}' for t in times]}")
