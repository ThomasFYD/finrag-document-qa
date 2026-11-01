# Verification record

Date: 2026-11-01. Environment: local macOS arm64 / Python 3.12.0.

- Dependency consistency and source compilation checks pass.
- All 17 unit, mocked-API, real-model/PDF integration, and demo tests passed. Reproduce them with `scripts/validate.py`.
- The source PDF has 9 pages; the baseline has 22 chunks and the semantic index has 65, each embedded in 384 dimensions.
- Model revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Development and test evaluations ran against the real source PDF. Per-question results are in `reports/`.
- The validation script explicitly enables Hugging Face / Transformers offline mode and makes no DeepInfra calls.
- The local Streamlit app displays actual retrieved evidence, page references, table context, and extraction-quality notices.
- Dependency/model downloads require internet access; subsequent local indexing, retrieval, and evaluation work offline.

Not verified: live DeepInfra requests, HyDE effectiveness, LLaMA/Mistral answer quality, actual API billing, large-scale/high-concurrency performance, or public deployment.

See README and `scripts/validate.py` for reproduction. Mock transport tests are not live-service validation.
