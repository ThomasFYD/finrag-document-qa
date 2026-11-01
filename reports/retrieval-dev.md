# Local retrieval evaluation — measured results

Split: `dev` · top-k: 5 · 2026-11-01T07:15:53.670932+00:00

| Pipeline | Answerable n | Hit@k | All-evidence coverage | Evidence Recall@k | MRR@k | Warm p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_dense | 10 | 60.0% | 60.0% | 60.0% | 0.367 | 0.3 / 0.5 |
| semantic_dense | 10 | 80.0% | 80.0% | 80.0% | 0.675 | 0.3 / 0.3 |
| semantic_hybrid | 10 | 90.0% | 90.0% | 90.0% | 0.567 | 0.6 / 1.2 |

Structured semantic vs. baseline all-evidence coverage: **+20.0 percentage points**.
Relative change: **+33.3%**.

HyDE and remote answer quality have not been tested live. These results do not support a claim of 15% improvement from HyDE.

Each question has one or more required evidence groups, with alternative valid sources. A retrieved chunk must contain the specified facts and context to match a group.
Hit@k requires at least one group; all-evidence coverage requires every group; MRR measures the position of the first matching group.
Unanswerable questions are excluded from these denominators. Their retrieval scores are retained for analysis; similarity is not confidence.

Latency is measured with query embeddings already cached, including retrieval and packing but excluding model loading and first-time encoding. It is not an online SLA.
A single report, hand-authored questions, and automatic substring checks do not establish generalization or generated-answer correctness. Manual review is still needed.

## Questions without complete evidence coverage

- `baseline_dense` · `D04` · 下次现金股息每股多少，什么时候支付，股权登记日是哪天？ (evidence-group ranks: [None])
- `baseline_dense` · `D05` · What was net cash provided by operating activities for the six months ended July 28, 2024? (evidence-group ranks: [None])
- `baseline_dense` · `D06` · 第二季度 GAAP 净利润是多少？ (evidence-group ranks: [None])
- `baseline_dense` · `D07` · What were Q2 FY2025 GAAP operating expenses? (evidence-group ranks: [None])
- `semantic_dense` · `D04` · 下次现金股息每股多少，什么时候支付，股权登记日是哪天？ (evidence-group ranks: [None])
- `semantic_dense` · `D10` · What is NVIDIA’s expected Q3 FY2025 tax rate and range? (evidence-group ranks: [None])
- `semantic_hybrid` · `D04` · 下次现金股息每股多少，什么时候支付，股权登记日是哪天？ (evidence-group ranks: [None])

## Unanswerable questions: highest retrieved score (not confidence)

- `baseline_dense` · `N01` · cosine=0.893
- `baseline_dense` · `N02` · cosine=0.780
- `semantic_dense` · `N01` · cosine=0.896
- `semantic_dense` · `N02` · cosine=0.781
- `semantic_hybrid` · `N01` · cosine=0.890
- `semantic_hybrid` · `N02` · cosine=0.781
