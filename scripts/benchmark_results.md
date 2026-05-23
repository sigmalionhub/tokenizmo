### Compression (chars per token — higher is better)

| Tokenizer          | Vocab  |     EN |     RU |   Code | RU/EN ratio |
|:-------------------|-------:|------:|------:|------:|------------:|
| tiktoken cl100k    |   100k |   5.54 |   2.29 |   4.03 |       2.039x |
| tiktoken o200k     |   200k |   5.58 |   4.21 |   4.04 |       1.115x |
| tokenismo v6       |   262k |   5.69 |   5.80 |   4.09 |       0.827x |

### Throughput (MB/s on 64 KB input — higher is better)

| Tokenizer          |       EN |       RU |     Code | vs o200k (EN) |
|:-------------------|---------:|---------:|---------:|---------------:|
| tiktoken cl100k    |      8.8 |      8.1 |      5.4 |           0.7x |
| tiktoken o200k     |     13.5 |      8.1 |      6.3 |           1.0x |
| tokenismo v6       |     76.0 |     92.0 |     55.8 |           5.6x |

### tokenismo v6 vs tiktoken o200k (direct comparison)

- **EN**: 5.69 vs 5.58 cpt  (+2.1%)  ✅ better
- **RU**: 5.80 vs 4.21 cpt  (+37.7%)  ✅ better
- **Code**: 4.09 vs 4.04 cpt  (+1.3%)  ✅ better