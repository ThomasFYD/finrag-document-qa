# RAG bottlenecks and targeted questions

These questions guide the implementation and evaluation of the main pipeline rather than replacing it with a survey.

| Bottleneck | Targeted financial-report question | Implemented mitigation | Remaining work |
|---|---|---|---|
| Extraction fidelity | What units apply to 30,040? | Layout extraction, source PDF, per-page text, hashes | OCR, merged cells, reliable footnote-table reconstruction |
| Lost headers | Does 31,480 refer to a quarter or six months? | Repeated headers/units and cross-page header provenance | General table structure and cross-page row labels |
| Chunk granularity | Why does a title-only chunk rank first? | Token bounds, semantic boundaries, fixed baseline | Minimum useful-content rules, parent/child retrieval, ablations |
| Cross-language matching | Does a Chinese Gaming question retrieve English evidence? | Multilingual E5 with shared vector space | Cross-language reranking and query rewriting |
| Conflicting periods | Can Q2 actuals be confused with Q3 guidance or FY2026? | Explicit period metadata and generation instructions | Structured date filters, entity parsing, conflict checks |
| Exact terms and numbers | Can dense retrieval miss precise identifiers? | BM25/dense rank fusion | Candidate/weight tuning and multilingual lexical strategy |
| Multiple facts | Can a comparison retrieve both Gaming and Automotive? | Multiple evidence groups and all-evidence coverage | Query decomposition and controlled calculations |
| Unsupported questions | Why does an absent headcount still get a high score? | Evidence-only mode, separately reported negatives | Calibrated answerability and refusal evaluation |
| HyDE drift | Can an invented amount leak into the answer? | Draft used only for retrieval, never as final evidence | Live quality/cost tests and fallback policy |
| Excess context | Does larger top-k add evidence or another quarter? | Whole-chunk packing, deduplication, size budget | Provider-specific token counting and reranking |
| Cited hallucinations | Can a valid citation still support a wrong amount? | JSON/source-ID checks and conservative error display | Atomic-fact, unit, and period validation; human review |
| Stale documents | Could an updated PDF retain old vectors? | Source/config checks and versioned atomic index promotion | Incremental updates, deletion, scheduled refresh |
| Performance and cost | Is time spent loading, encoding, searching, or generating? | Cache, retrieval timings, API usage/latency records | Cold/warm separation, concurrency testing, SLOs, spend limits |
| Prompt injection | What if a document tells the model to reveal secrets? | Evidence treated as data; no generation tools; output validation | Live adversarial evaluation; mitigations are not a complete defense |
| Access control | Could one user retrieve another user's documents? | Current local-only single-user scope | Authentication, pre-retrieval ACLs, tenant isolation, audit |
| Scale | Can exact search handle millions of chunks? | Inspectable NumPy index for this small corpus | ANN/vector DB, capacity planning, recall/latency trade-offs |

## Three observations worth discussing

1. **Retrieving a number is not sufficient.** Its unit, period, row label, and accounting basis must survive.
2. **A high retrieval score does not establish answerability.** Current negative examples also score highly, so an arbitrary 0.7 cutoff would be misleading.
3. **Claims must follow experiment boundaries.** Layout preservation, semantic chunking, and hybrid search need separate ablations; the combined gain cannot be attributed to HyDE.

## Proposed next experiments

These are future work, not implemented claims:

- Add another quarter and company, including identically named but conflicting financial metrics.
- Add a structure-preserving fixed-size intermediate baseline to isolate semantic-boundary effects.
- Build an independent held-out set covering merged headers, cross-page rows, calculation questions, stale facts, and adversarial instructions.
- With API access, compare dense/HyDE retrieval first, then LLaMA/Mistral generation on fixed evidence.
- Choose rerankers, OCR, or a database based on measured failure modes rather than adding tools for their own sake.
