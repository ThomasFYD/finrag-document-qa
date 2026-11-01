# Local retrieval evaluation — measured results

Split: `test` · top-k: 5 · 2026-11-01T07:16:02.271686+00:00

| Pipeline | Answerable n | Hit@k | All-evidence coverage | Evidence Recall@k | MRR@k | Warm p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_dense | 22 | 59.1% | 59.1% | 59.1% | 0.350 | 0.3 / 0.4 |
| semantic_dense | 22 | 81.8% | 81.8% | 81.8% | 0.602 | 0.3 / 0.4 |
| semantic_hybrid | 22 | 90.9% | 90.9% | 90.9% | 0.632 | 0.7 / 1.0 |

Structured semantic vs. baseline all-evidence coverage: **+22.7 percentage points**.
Relative change: **+38.5%**.

HyDE and remote answer quality have not been tested live. These results do not support a claim of 15% improvement from HyDE.

Each question has one or more required evidence groups, with alternative valid sources. A retrieved chunk must contain the specified facts and context to match a group.
Hit@k requires at least one group; all-evidence coverage requires every group; MRR measures the position of the first matching group.
Unanswerable questions are excluded from these denominators. Their retrieval scores are retained for analysis; similarity is not confidence.

Latency is measured with query embeddings already cached, including retrieval and packing but excluding model loading and first-time encoding. It is not an online SLA.
A single report, hand-authored questions, and automatic substring checks do not establish generalization or generated-answer correctness. Manual review is still needed.

## Questions without complete evidence coverage

- `baseline_dense` · `T01` · 游戏业务第二季度收入及同比增长率分别是多少？ (evidence-group ranks: [None])
- `baseline_dense` · `T06` · What was GAAP net income for the six months ended July 28, 2024? (evidence-group ranks: [None])
- `baseline_dense` · `T07` · 2024 年 7 月 28 日，现金、现金等价物及有价证券总额是多少？ (evidence-group ranks: [None])
- `baseline_dense` · `T09` · 前六个月自由现金流是多少？ (evidence-group ranks: [None])
- `baseline_dense` · `T13` · How much net cash did operating activities provide in the three months ended July 28, 2024? (evidence-group ranks: [None])
- `baseline_dense` · `T15` · What was research and development expense for the three months ended July 28, 2024? (evidence-group ranks: [None])
- `baseline_dense` · `T16` · 截至 2024 年 7 月 28 日的六个月所得税费用是多少？ (evidence-group ranks: [None])
- `baseline_dense` · `T18` · What was cash paid for income taxes, net, for the three months ended July 28, 2024? (evidence-group ranks: [None])
- `baseline_dense` · `T19` · In the Q3 FY2025 outlook reconciliation table, what are GAAP and non-GAAP operating expenses? (evidence-group ranks: [None, None])
- `semantic_dense` · `T04` · 截至 2024 年 7 月 28 日的六个月收入是多少？不要季度收入。 (evidence-group ranks: [None])
- `semantic_dense` · `T07` · 2024 年 7 月 28 日，现金、现金等价物及有价证券总额是多少？ (evidence-group ranks: [None])
- `semantic_dense` · `T16` · 截至 2024 年 7 月 28 日的六个月所得税费用是多少？ (evidence-group ranks: [None])
- `semantic_dense` · `T21` · What calendar date ended NVIDIA’s second quarter of fiscal 2025? (evidence-group ranks: [None])
- `semantic_hybrid` · `T04` · 截至 2024 年 7 月 28 日的六个月收入是多少？不要季度收入。 (evidence-group ranks: [None])
- `semantic_hybrid` · `T21` · What calendar date ended NVIDIA’s second quarter of fiscal 2025? (evidence-group ranks: [None])

## Unanswerable questions: highest retrieved score (not confidence)

- `baseline_dense` · `N03` · cosine=0.867
- `baseline_dense` · `N04` · cosine=0.843
- `baseline_dense` · `N05` · cosine=0.750
- `baseline_dense` · `N06` · cosine=0.806
- `semantic_dense` · `N03` · cosine=0.872
- `semantic_dense` · `N04` · cosine=0.837
- `semantic_dense` · `N05` · cosine=0.747
- `semantic_dense` · `N06` · cosine=0.800
- `semantic_hybrid` · `N03` · cosine=0.872
- `semantic_hybrid` · `N04` · cosine=0.837
- `semantic_hybrid` · `N05` · cosine=0.747
- `semantic_hybrid` · `N06` · cosine=0.798
