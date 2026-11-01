# Optional DeepInfra and HyDE integration

**Implemented, not live-validated.** Live validation is deferred until API credentials are configured. No paid API requests have been performed.

## Implemented behavior

- `remote.py` calls `https://api.deepinfra.com/v1/openai/chat/completions`.
- Model IDs are checked against the current provider catalog rather than guessed or hardcoded as defaults.
- HTTP 429/5xx responses have finite retries. Timeouts are not automatically retried to reduce duplicate-billing risk.
- Responses record model, token usage, latency, finish reason, and cost estimates when verified rates are supplied.
- `pipeline.py` implements hypothetical-passage generation, passage embedding, retrieval of real evidence, and optional grounded answering.
- `evaluation.py` includes a HyDE retrieval comparison and a multi-model generation benchmark.
- Transport, response parsing, and citation validation are covered by mock-based tests. These do not establish live compatibility, quality, or actual cost.

## Local configuration

Copy `.env.example` to `.env`, then enter `DEEPINFRA_API_KEY` and `DEEPINFRA_MODEL` in your local editor. Never paste the key into a chat or commit it. `.env` is ignored by version control.

Choose exact current model IDs from the [DeepInfra catalog](https://deepinfra.com/models). Available LLaMA/Mistral versions, prices, and supported features change; the project deliberately leaves them unconfigured.

Populate the `models` array in `config/remote_models.json` for comparisons:

```json
{
  "id": "exact-current-provider-model-id",
  "label": "descriptive model label",
  "input_usd_per_million": null,
  "output_usd_per_million": null,
  "pricing_verified_on": null
}
```

Replace null rates only after verifying pricing. Estimates use reported input/output tokens and those rates; they are not billing guarantees and may omit unreported charges.

## Bounded smoke tests

These commands make paid requests once credentials are configured:

```bash
.venv/bin/finrag ask "What was Q2 FY2025 revenue?" --generate
.venv/bin/finrag ask "What was Q2 FY2025 revenue?" --strategy hyde --allow-remote --generate
.venv/bin/finrag evaluate --split dev --hyde --allow-remote --limit 3
.venv/bin/finrag benchmark --allow-remote --limit 3
```

A benchmark permits 1–3 models and at most 10 questions per model. Each answer requests at most 700 output tokens; a HyDE draft requests at most 180. These are call/output bounds, not an account-level dollar budget.

Questions and selected evidence are sent to DeepInfra. Validate with the public sample report first; private documents require a separate organizational data-handling review.

## Live validation checklist

1. Confirm the chosen IDs are available and accept the required request parameters.
2. Inspect actual JSON, finish reason, and usage fields, as well as timeout/rate-limit behavior.
3. Verify amounts, units, accounting basis, periods, and whether citations support each claim.
4. Compare LLaMA/Mistral on the same questions and retrieved evidence, measuring correctness, faithfulness, latency, and estimated cost.
5. Compare direct retrieval against HyDE, including cases where a misleading draft degrades retrieval.

Human quality scores remain null until reviewed. Do not claim completed LLaMA/Mistral benchmarks or a 15% HyDE improvement before these experiments run.
