"""Evidence-span evaluation independent of chunk IDs, with transparent denominators."""
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import statistics
import time

import numpy as np

from .pipeline import RAGPipeline


def norm(text):
    return re.sub(r"[\W_]", "", text.casefold())


def matches(alternative, item):
    return (item["metadata"]["doc_id"] == alternative["doc_id"]
            and item["metadata"]["page"] in alternative["pages"]
            and all(norm(phrase) in norm(item["text"]) for phrase in alternative["contains"]))


def evidence_metrics(question, evidence):
    groups = question["evidence"]
    if not groups:
        return None
    ranks = []
    for group in groups:
        rank = next((i for i, item in enumerate(evidence, 1)
                     if any(matches(a, item) for a in group["alternatives"])), None)
        ranks.append(rank)
    found = [r for r in ranks if r is not None]
    return {"hit": int(bool(found)), "all_evidence": int(len(found) == len(groups)),
            "evidence_recall": len(found) / len(groups),
            "reciprocal_rank": 1 / min(found) if found else 0, "ranks": ranks}


def load_questions(root, split="test", limit=None):
    source = root / "evals/questions.json"
    dataset = json.loads(source.read_text())
    questions = [q for q in dataset["questions"] if split == "all" or q["split"] == split]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("The evaluation set is empty.")
    return questions, hashlib.sha256(source.read_bytes()).hexdigest()


def evaluate(settings, embeddings, split="test", k=5, remote=None, limit=None):
    questions, digest = load_questions(settings.root, split, limit)
    configs = [("baseline_dense", "baseline", "dense"), ("semantic_dense", "semantic", "dense"),
               ("semantic_hybrid", "semantic", "hybrid")]
    if remote:
        configs.append(("semantic_hyde", "semantic", "hyde"))
    rows, summary, manifests = [], {}, {}
    for label, method, strategy in configs:
        pipeline = RAGPipeline(settings, embeddings, method, remote)
        manifests[label] = pipeline.index.manifest
        # Explicit warm-up; results label latency as warm-cache, not cold-start or server throughput.
        for question in questions:
            embeddings.embed_query(question["question"])
        variant = []
        for question in questions:
            result = pipeline.ask(question["question"], k=k, strategy=strategy)
            # Evaluate returned evidence after context packing, not every stored chunk.
            metrics = evidence_metrics(question, result["evidence"])
            row = {"id": question["id"], "question": question["question"], "category": question["category"],
                "variant": label, "answerable": question["answerable"], "metrics": metrics,
                "retrieval_ms": result["retrieval_ms"], "context_bytes": result["context_bytes_used"],
                "top_cosine": result["evidence"][0]["cosine"] if result["evidence"] else None,
                "retrieved": [{"chunk_id": s["metadata"]["chunk_id"], "page": s["metadata"]["page"],
                               "text": s["text"], "cosine": s["cosine"]} for s in result["evidence"]],
                "api_calls": result["api_calls"]}
            rows.append(row); variant.append(row)
        supported = [r for r in variant if r["answerable"]]
        timing = [r["retrieval_ms"] for r in variant]
        summary[label] = {"answerable_n": len(supported), "unanswerable_n": len(variant) - len(supported),
            **{key: statistics.mean(r["metrics"][key] for r in supported) if supported else None
               for key in ["hit", "all_evidence", "evidence_recall", "reciprocal_rank"]},
            "warm_p50_ms": float(np.percentile(timing, 50)), "warm_p95_ms": float(np.percentile(timing, 95)),
            "mean_context_bytes": statistics.mean(r["context_bytes"] for r in variant)}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "split": split, "k": k,
            "dataset_sha256": digest, "machine": platform.platform(), "embedding": embeddings.fingerprint,
            "latency_protocol": "query embeddings warmed; excludes model loading, includes retrieval and packing; HyDE includes live generation",
            "limitations": ["Small, hand-authored single-report dataset; no generalization claim.",
                             "Substring evidence checks are proxies, not LLM answer correctness or faithfulness.",
                             "Unanswerable questions excluded from retrieval recall; scores reported, no calibrated rejection threshold."],
            "summary": summary, "indexes": manifests, "rows": rows}


def report_markdown(report):
    lines = ["# Local retrieval evaluation — measured results", "", f"Split: `{report['split']}` · top-k: {report['k']} · {report['created_at']}", "",
             "| Pipeline | Answerable n | Hit@k | All-evidence coverage | Evidence Recall@k | MRR@k | Warm p50 / p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, s in report["summary"].items():
        lines.append(f"| {label} | {s['answerable_n']} | {s['hit']:.1%} | {s['all_evidence']:.1%} | {s['evidence_recall']:.1%} | {s['reciprocal_rank']:.3f} | {s['warm_p50_ms']:.1f} / {s['warm_p95_ms']:.1f} |")
    base = report["summary"]["baseline_dense"]["all_evidence"]
    semantic = report["summary"]["semantic_dense"]["all_evidence"]
    lines.extend(["", f"Structured semantic vs. baseline all-evidence coverage: **{(semantic - base) * 100:+.1f} percentage points**.",
                  f"Relative change: **{(semantic / base - 1):+.1%}**." if base else "Relative change is undefined because the baseline is zero.",
                  "", "HyDE and remote answer quality have not been tested live. These results do not support a claim of 15% improvement from HyDE." if "semantic_hyde" not in report["summary"] else "HyDE retrieval results are included above; answer quality requires separate review.",
                  "", "Each question has one or more required evidence groups, with alternative valid sources. A retrieved chunk must contain the specified facts and context to match a group.",
                  "Hit@k requires at least one group; all-evidence coverage requires every group; MRR measures the position of the first matching group.",
                  "Unanswerable questions are excluded from these denominators. Their retrieval scores are retained for analysis; similarity is not confidence.",
                  "", "Latency is measured with query embeddings already cached, including retrieval and packing but excluding model loading and first-time encoding. It is not an online SLA.",
                  "A single report, hand-authored questions, and automatic substring checks do not establish generalization or generated-answer correctness. Manual review is still needed.",
                  "", "## Questions without complete evidence coverage", ""])
    for row in report["rows"]:
        if row["metrics"] and not row["metrics"]["all_evidence"]:
            lines.append(f"- `{row['variant']}` · `{row['id']}` · {row['question']} (evidence-group ranks: {row['metrics']['ranks']})")
    lines += ["", "## Unanswerable questions: highest retrieved score (not confidence)", ""]
    for row in report["rows"]:
        if not row["answerable"]:
            score = f"{row['top_cosine']:.3f}" if row['top_cosine'] is not None else "no evidence"
            lines.append(f"- `{row['variant']}` · `{row['id']}` · cosine={score}")
    return "\n".join(lines) + "\n"


def benchmark_remote(settings, embeddings, models, limit=3):
    if not models or not 1 <= limit <= 10 or len(models) > 3:
        raise ValueError("Configure 1–3 models and 1–10 questions per model.")
    from .remote import DeepInfraClient
    questions, digest = load_questions(settings.root, "test", limit)
    results = []
    for model in models:
        client = DeepInfraClient(model=model["id"], pricing=model)
        try:
            client.validate_model()
            pipeline = RAGPipeline(settings, embeddings, "semantic", client)
            for question in questions:
                started = time.perf_counter()
                result = pipeline.ask(question["question"], generate=True)
                # Human scoring is deliberately left blank, never invented from citation validity.
                results.append({"model": model["id"], "question_id": question["id"],
                    "question": question["question"], "reference_answer": question["reference_answer"],
                    "result": result, "total_ms": (time.perf_counter() - started) * 1000,
                    "human_correctness_0_to_2": None, "human_faithfulness_0_to_2": None,
                    "pricing": model})
        finally:
            client.close()
    return {"created_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": digest,
            "protocol": "Same questions, same semantic dense retrieval, temperature 0; quality awaits manual scoring.", "rows": results}
