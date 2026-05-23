### Compression (chars per token — higher is better)

| Tokenizer          | Vocab  |     EN |     RU |   Code | RU/EN ratio |
|:-------------------|-------:|------:|------:|------:|------------:|
| tiktoken cl100k    |   100k |   5.54 |   2.29 |   4.03 |       2.039x |
| tiktoken o200k     |   200k |   5.58 |   4.21 |   4.04 |       1.115x |
| XLM-R (250k)       |   250k |   4.50 |   4.25 |   3.10 |       0.890x |
| mBERT (120k)       |   120k |   4.80 |   3.73 |   2.50 |       1.084x |
| tokenismo v6       |   262k |   5.69 |   5.80 |   4.09 |       0.827x |

### Throughput (MB/s on 64 KB input — higher is better)

| Tokenizer          |       EN |       RU |     Code | vs o200k (EN) |
|:-------------------|---------:|---------:|---------:|---------------:|
| tiktoken cl100k    |      9.9 |      9.4 |      6.7 |           0.6x |
| tiktoken o200k     |     16.8 |     12.9 |      9.7 |           1.0x |
| XLM-R (250k)       |      2.0 |      3.1 |      2.2 |           0.1x |
| mBERT (120k)       |      2.3 |      3.0 |      1.9 |           0.1x |
| tokenismo v6       |     98.8 |    138.1 |     86.1 |           5.9x |

### tokenismo v6 vs tiktoken o200k (direct comparison)

- **EN**: 5.69 vs 5.58 cpt  (+2.1%)  ✅ better
- **RU**: 5.80 vs 4.21 cpt  (+37.7%)  ✅ better
- **Code**: 4.09 vs 4.04 cpt  (+1.3%)  ✅ better