# Model Configs

All configs are Llama-style decoder-only models (RMSNorm, SwiGLU, RoPE, untied embeddings) with the
GPT-NeoX/OLMo tokenizer (`vocab_size = 50280`) and a 2048-token context. Pass any file to a training
entrypoint with `--model-config configs/<name>.json`.

*Aspect ratio* = hidden / layers. *Backbone* counts transformer blocks only; *Total* adds the input
embedding and LM head.

## Training budgets

Every run uses global batch size 512 × 2048 tokens (≈1M tokens/step) and a cosine schedule with 10% warmup.

| Tier | Tokens | Steps |
|---|---:|---:|
| 200M | 4B | 3.8k |
| 300M | 6B | 5.7k |
| 400M | 8B | 7.6k |
| 500M | 10B | 9.5k |
| 1.6B | 32B | 30.4k |

## Shape design

Unless noted otherwise, shapes follow:

```text
heads        = 16
head_dim     = even integer
hidden       = heads * head_dim
intermediate = ceil(hidden * 8/3, multiple_of=16)
```

## 400M: fixed total size (405M ± 3%)

| Config | Layers | Hidden | Intermediate | head_dim | Aspect ratio | Backbone | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama_400m_L16.json` | 16 | 1216 | 3248 | 76 | 76.00 | 284M | 407M |
| `llama_400m_L20.json` | 20 | 1120 | 2992 | 70 | 56.00 | 301M | 414M |
| `llama_400m_L24.json` (base) | 24 | 1024 | 2736 | 64 | 42.67 | 302M | 405M |
| `llama_400m_L28.json` | 28 | 960 | 2560 | 60 | 34.29 | 310M | 406M |
| `llama_400m_L32.json` | 32 | 896 | 2400 | 56 | 28.00 | 309M | 399M |
| `llama_400m_L42.json` | 42 | 800 | 2144 | 50 | 19.05 | 324M | 404M |
| `llama_400m_L70.json` | 70 | 640 | 1712 | 40 | 9.14 | 345M | 409M |

## 300M backbone: fixed backbone size (302M ± 3%)

| Config | Layers | Hidden | Intermediate | head_dim | Aspect ratio | Backbone | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama_300m_backbone_L16.json` | 16 | 1248 | 3328 | 78 | 78.00 | 299M | 425M |
| `llama_300m_backbone_L24.json` | 24 | 1024 | 2736 | 64 | 42.67 | 302M | 405M |
| `llama_300m_backbone_L32.json` | 32 | 896 | 2400 | 56 | 28.00 | 309M | 399M |
| `llama_300m_backbone_L42.json` | 42 | 768 | 2048 | 48 | 18.29 | 297M | 375M |
| `llama_300m_backbone_L70.json` | 70 | 608 | 1632 | 38 | 8.69 | 312M | 373M |
| `llama_300m_backbone_L84.json` | 84 | 544 | 1456 | 34 | 6.48 | 299M | 354M |

L24 and L32 are the same shapes as `llama_400m_L24.json` and `llama_400m_L32.json`.

## Depth scaling ladder

Three aspect ratios (≈28, ≈43, ≈75) at each size, with total size held within ±3.5% of the middle shape.

| Config | Layers | Hidden | Intermediate | head_dim | Aspect ratio | Backbone | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama_200m_L24.json` | 24 | 672 | 1792 | 42 | 28.00 | 130M | 198M |
| `llama_200m_L18.json` | 18 | 768 | 2048 | 48 | 42.67 | 127M | 205M |
| `llama_200m_L12.json` | 12 | 896 | 2400 | 56 | 74.67 | 116M | 206M |
| `llama_300m_L28.json` | 28 | 800 | 2144 | 50 | 28.57 | 216M | 296M |
| `llama_300m_L21.json` | 21 | 896 | 2400 | 56 | 42.67 | 203M | 293M |
| `llama_300m_L14.json` | 14 | 1056 | 2816 | 66 | 75.43 | 187M | 294M |
| `llama_400m_L32.json` | 32 | 896 | 2400 | 56 | 28.00 | 309M | 399M |
| `llama_400m_L24.json` | 24 | 1024 | 2736 | 64 | 42.67 | 302M | 405M |
| `llama_400m_L16.json` | 16 | 1216 | 3248 | 76 | 76.00 | 284M | 407M |
| `llama_500m_L34.json` | 34 | 992 | 2656 | 62 | 29.18 | 403M | 502M |
| `llama_500m_L26.json` | 26 | 1120 | 2992 | 70 | 43.08 | 392M | 504M |
| `llama_500m_L17.json` | 17 | 1344 | 3584 | 84 | 79.06 | 368M | 504M |

## 1.6B

The base shape (L28) matches Qwen3-1.7B, using grouped-query attention with 16 query and 8 KV heads and
`intermediate = 3 * hidden`.

| Config | Layers | Hidden | Intermediate | head_dim | Aspect ratio | Backbone | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama_1600m_L28.json` (base) | 28 | 2048 | 6144 | 128 | 73.14 | 1.409B | 1.615B |
| `llama_1600m_L40.json` | 40 | 1728 | 5184 | 108 | 43.20 | 1.433B | 1.607B |
| `llama_1600m_L54.json` | 54 | 1504 | 4512 | 94 | 27.85 | 1.466B | 1.617B |
