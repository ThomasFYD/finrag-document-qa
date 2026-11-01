"""Small CLI: doctor, extract, index, ask, evaluate, benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys

from .config import Settings, ROOT


def parser():
    p = argparse.ArgumentParser(description="FinRAG: local financial retrieval and grounded QA")
    p.add_argument("--root", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect local setup without paid API calls")
    sub.add_parser("extract", help="Extract PDF text with page references")
    sub.add_parser("index", help="Build baseline and semantic indexes")
    q = sub.add_parser("ask", help="Ask a question; return evidence by default")
    q.add_argument("question")
    q.add_argument("--method", choices=["baseline", "semantic"], default="semantic")
    q.add_argument("--strategy", choices=["dense", "hybrid", "hyde"], default="dense")
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--doc-id")
    q.add_argument("--generate", action="store_true", help="Enable paid DeepInfra answer generation")
    q.add_argument("--allow-remote", action="store_true", help="Allow paid HyDE requests")
    q.add_argument("--json", action="store_true")
    e = sub.add_parser("evaluate", help="Evaluate retrieval without LLM calls by default")
    e.add_argument("--split", choices=["dev", "test", "all"], default="test")
    e.add_argument("--top-k", type=int, default=5)
    e.add_argument("--hyde", action="store_true")
    e.add_argument("--allow-remote", action="store_true")
    e.add_argument("--limit", type=int)
    b = sub.add_parser("benchmark", help="Compare DeepInfra generation models")
    b.add_argument("--allow-remote", action="store_true", required=True)
    b.add_argument("--limit", type=int, default=3)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = Settings.load(args.root)
        if args.command == "doctor":
            print(json.dumps({"python": sys.version.split()[0], "project": str(settings.root),
                "model_downloaded": (settings.model_path / "config.json").exists(),
                "embedding_revision": settings.model_revision,
                "indexes": {m: (settings.root / "data/indexes" / m / "CURRENT").exists() for m in ["baseline", "semantic"]},
                "deepinfra_key_configured": settings.remote_enabled,
                "deepinfra_model_configured": bool(os.getenv("DEEPINFRA_MODEL"))}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "extract":
            from .ingest import read_corpus
            pages, sources = read_corpus(settings.root)
            directory = settings.root / "data/extracted"
            directory.mkdir(parents=True, exist_ok=True)
            for page in pages:
                (directory / f"{page.metadata['doc_id']}-p{page.metadata['page']:03}.txt").write_text(page.page_content)
            print(f"Extracted {len(pages)} pages to {directory}")
            return 0
        # Fail fast before loading the embedding model if remote access was not enabled.
        remote = None
        wants_hyde = (args.command == "ask" and args.strategy == "hyde") or (args.command == "evaluate" and args.hyde)
        if wants_hyde and not args.allow_remote:
            raise ValueError("HyDE uses a paid API; explicitly pass --allow-remote.")
        if wants_hyde or (args.command == "ask" and args.generate):
            from .remote import DeepInfraClient
            remote = DeepInfraClient()
            remote.validate_model()
        from .embeddings import LocalEmbeddings
        embeddings = LocalEmbeddings(settings)
        try:
            if args.command == "index":
                from .index import build_indexes
                result = build_indexes(settings, embeddings)
                print(json.dumps({m: {"pages": r["pages"], "chunks": r["chunks"], "dimension": r["dimension"]} for m, r in result.items()}, indent=2))
            elif args.command == "ask":
                from .pipeline import RAGPipeline
                result = RAGPipeline(settings, embeddings, args.method, remote).ask(args.question,
                    k=args.top_k, strategy=args.strategy, generate=args.generate, doc_id=args.doc_id)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(result["answer"] or "Retrieved source evidence (not a generated answer; similarity is not confidence):")
                    for s in result["evidence"]:
                        print(f"\n[{s['citation']}] PDF page {s['metadata']['page']} · cosine {s['cosine']:.3f}\n{s['text']}")
                    print(f"\nRetrieval time {result['retrieval_ms']:.1f} ms · context {result['context_bytes_used']} bytes")
            elif args.command == "evaluate":
                from .evaluation import evaluate, report_markdown
                if args.hyde and (not args.limit or not 1 <= args.limit <= 10):
                    raise ValueError("Set --limit 1..10 for HyDE evaluation to bound API requests.")
                result = evaluate(settings, embeddings, args.split, args.top_k, remote, args.limit)
                directory = settings.root / "reports"
                directory.mkdir(exist_ok=True)
                stem = f"retrieval-{args.split}" + ("-hyde" if args.hyde else "")
                (directory / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                (directory / f"{stem}.md").write_text(report_markdown(result))
                print(report_markdown(result))
            elif args.command == "benchmark":
                from .evaluation import benchmark_remote
                models = json.loads((settings.root / "config/remote_models.json").read_text())["models"]
                result = benchmark_remote(settings, embeddings, models, args.limit)
                path = settings.root / "reports/deepinfra-benchmark.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"Saved model comparison to {path}; human quality scores remain pending.")
        finally:
            if remote:
                remote.close()
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
