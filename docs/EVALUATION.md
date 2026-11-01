# Evaluation methodology

## Dataset and scope

The corpus is a single nine-page NVIDIA FY2025 Q2 earnings release. The 38 manually authored questions were saved before the first evaluation: development contains 10 answerable and 2 unanswerable questions; test contains 22 and 4. Generated model responses are not used as ground truth.

Questions cover English/Chinese retrieval, units, GAAP vs. non-GAAP, actual results vs. outlook, quarter vs. six-month periods, cross-page tables, multiple evidence groups, and unsupported requests. Direct paraphrases of the same fact do not cross splits. However, the test set contains related single-fact and comparison questions, so its samples are not statistically independent.

The implementer authored the dataset, and test failures have now been inspected. This is not an independent blind benchmark. Future tuning should use development data, followed by a new held-out dataset across companies and periods.

## Compared pipelines

- `baseline_dense`: fixed-size splitting and E5 dense retrieval.
- `semantic_dense`: structural context preservation, adjacent-sentence semantic boundaries, and E5 dense retrieval.
- `semantic_hybrid`: the same semantic index plus BM25/reciprocal rank fusion.

All variants share the embedding revision, corpus, top-k=5, total context byte budget, and scoring rules. The baseline's 360-token budget applies to its body, with a short document title added; the semantic budget includes repeated context. Both enforce the embedding model's 512-token total limit.

**This is a comparison of complete pipeline variants, not a single-variable semantic-chunking ablation.** To isolate the contribution of semantic boundaries, add an intermediate structure-preserving fixed-size variant and equalize final token budgets exactly.

## Scoring

Each answerable question specifies one or more required evidence groups. Alternatives within a group are OR conditions; separate groups are AND conditions. Matching uses normalized text containment and physical source page numbers, rather than fixed chunk IDs.

- **Hit@k:** at least one required group was retrieved.
- **All-evidence coverage:** every required group was retrieved.
- **Evidence Recall@k:** the fraction of required groups retrieved.
- **MRR@k:** the reciprocal rank of the first matched group.

Text matching is a proxy. It may detect numbers and labels without fully validating their semantic relationship. Manual review remains necessary.

Unanswerable questions have no required evidence groups and are excluded from answerable recall denominators. Their top retrieval scores are reported separately. No answerability threshold or refusal-accuracy claim is inferred from those scores.

## Timing protocol

Query embeddings are warmed before timed retrieval. Reported latency includes retrieval and context packing, but excludes model loading, first-time query encoding, and remote generation. It is not cold-start latency, end-to-end first-request latency, or a production SLA. The live UI may show substantially longer request times.

## Reproducing and extending experiments

1. Archive existing reports and record corpus hashes, embedding revision, chunking parameters, top-k, and context budget.
2. Change one justified variable on the development set.
3. Keep per-question failures and classify them as extraction, chunking, recall, ranking, context selection, or generation failures.
4. Freeze configuration and evaluate on a new held-out set. Add other companies and reporting periods.
5. When reporting an improvement, state whether it is a percentage-point difference or a relative change, and include raw numerators/denominators.

From 59.1% to 81.8% is approximately **+22.7 percentage points**, or **+38.5% relative**. These are evidence-coverage results for the current combined pipeline, not gains in generated-answer accuracy or measured HyDE performance.

## Generation evaluation — not yet executed

The benchmark harness fixes questions and retrieval settings and records model ID, answer, evidence, token usage, latency, and optional estimated cost. Missing prices remain null, not zero.

Review correctness and faithfulness independently on a 0–2 scale: 0 = incorrect/unsupported; 1 = partly correct/incomplete; 2 = complete and supported. Also inspect units, periods, source locations, and unsupported-question handling. Citation-ID validity alone is not faithfulness.

Evaluate HyDE separately as a retrieval change. Compare LLaMA/Mistral on the same real evidence. Context-window experiments should hold questions and retrieval candidates fixed, then vary the budget and measure completeness, errors, latency, and cost.
