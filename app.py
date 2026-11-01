"""Local Streamlit demo. Start with .venv/bin/streamlit run app.py."""
import json
import os

import streamlit as st

from finrag.config import Settings
from finrag.embeddings import LocalEmbeddings
from finrag.pipeline import RAGPipeline

st.set_page_config(page_title="FinRAG · Financial Document QA", page_icon="📑", layout="wide")
st.markdown("""<style>
.block-container {max-width:1180px;padding-top:2.6rem;}
.eyebrow {font-size:13px;letter-spacing:2px;color:#0d9488;font-weight:700;}
h1 {letter-spacing:-1.5px;}
div[data-testid="stMetric"] {background:#f3f7f8;border:1px solid #e0e9ea;border-radius:12px;padding:15px;}
div[data-testid="stMetricValue"] {font-size:26px;}
</style>""", unsafe_allow_html=True)
settings = Settings.load()


@st.cache_resource
def load_embeddings(revision):
    return LocalEmbeddings(settings)


@st.cache_resource
def load_pipeline(method, generation, revision):
    return RAGPipeline(settings, load_embeddings(revision), method)


st.markdown('<div class="eyebrow">FINRAG / DOCUMENT INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("Every number. Backed by evidence.")
st.write("Retrieve evidence from NVIDIA financial reports, compare retrieval methods, and trace each result to its source page.")

with st.sidebar:
    st.header("Retrieval settings")
    method = st.selectbox("Chunking method", ["semantic", "baseline"], format_func=lambda x: {"semantic": "Semantic + table context", "baseline": "Baseline: fixed-size chunks"}[x])
    options = ["dense", "hybrid"] + (["hyde"] if settings.remote_enabled and os.getenv("DEEPINFRA_MODEL") else [])
    strategy = st.selectbox("Search strategy", options, format_func=lambda x: {"dense": "Dense semantic search", "hybrid": "Hybrid: dense + keywords", "hyde": "HyDE (DeepInfra API)"}[x])
    k = st.slider("Maximum evidence chunks", 1, 10, 5)
    budget = st.slider("Evidence budget (UTF-8 bytes)", 3000, 24000, settings.context_bytes, step=1000)
    generate = st.toggle("Generate an answer with DeepInfra", value=False,
        disabled=not (settings.remote_enabled and os.getenv("DEEPINFRA_MODEL")))
    if generate or strategy == "hyde":
        st.caption("This request sends the question and relevant content to DeepInfra and incurs API charges.")
    else:
        st.caption("Local evidence retrieval. Runs offline after model download; no paid API calls.")
    st.divider()
    st.caption("Document: NVIDIA FY2025 Q2\n\nQuarter ended: July 28, 2024\n\nIndexes retain PDF pages, source hashes, and model versions.")

question_tab, results_tab, about_tab = st.tabs(["Ask the report", "Evaluation", "How it works"])
with question_tab:
    with st.form("question"):
        question = st.text_input("What would you like to know?", value="What was NVIDIA Q2 FY2025 revenue, and what are the units?", max_chars=2000)
        submit = st.form_submit_button("Find evidence", type="primary", use_container_width=True)
    st.caption("Try: What was six-month revenue? · 游戏业务收入同比增长多少？ · What is the Q3 revenue outlook?")
    if submit:
        try:
            with st.spinner("Finding relevant passages…"):
                generation = (settings.root / "data/indexes" / method / "CURRENT").read_text()
                pipeline = load_pipeline(method, generation, settings.model_revision)
                remote = None
                if generate or strategy == "hyde":
                    from finrag.remote import DeepInfraClient
                    remote = DeepInfraClient()
                    remote.validate_model()
                    pipeline = RAGPipeline(settings, load_embeddings(settings.model_revision), method, remote)
                try:
                    result = pipeline.ask(question, k=k, strategy=strategy, generate=generate, context_bytes=budget)
                finally:
                    if remote:
                        remote.close()
                st.session_state["result"] = result
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
            st.session_state.pop("result", None)
    if result := st.session_state.get("result"):
        st.subheader(result["question"])
        a, b, c = st.columns(3)
        a.metric("Evidence chunks", len(result["evidence"]))
        b.metric("Retrieval time", f"{result['retrieval_ms']:.0f} ms")
        c.metric("Context size", f"{result['context_bytes_used']:,} B")
        if result["answer"]:
            st.write(result["answer"])
        else:
            st.info("These are source excerpts, not a generated answer. Similarity does not establish answerability: check periods, units, and table headers.")
        if result.get("hyde_draft"):
            with st.expander("HyDE draft (retrieval aid, not evidence)"):
                st.write(result["hyde_draft"])
        for evidence in result["evidence"]:
            m = evidence["metadata"]
            with st.container(border=True):
                st.markdown(f"**[{evidence['citation']}] {m['title']} · Page {m['page']}**")
                st.caption(f"cosine {evidence['cosine']:.3f} · {m.get('kind', 'fixed')} · {m['tokens']} tokens · chunk {m['chunk_id'][:8]}")
                header_pages = m.get("header_pages", [])
                if header_pages and header_pages != [m["page"]]:
                    st.caption(f"Table headers from pages {', '.join(map(str, header_pages))}; verify both headers and data pages.")
                if m.get("quality_warning"):
                    st.warning(m["quality_warning"])
                st.text(evidence["text"])
        st.download_button("Export retrieval record", json.dumps(result, ensure_ascii=False, indent=2), "retrieval.json", "application/json")
    else:
        st.info("Ask a question to begin. Inspect the source excerpts, then compare baseline and semantic retrieval.")
    pdf = settings.root / "nvidia-fy2025-q2-earnings-release.pdf"
    if pdf.exists():
        st.download_button("Download source PDF (9 pages)", pdf.read_bytes(), pdf.name, "application/pdf")

with results_tab:
    report = settings.root / "reports/retrieval-test.json"
    if report.exists():
        data = json.loads(report.read_text())
        st.subheader("One dataset. Three retrieval pipelines.")
        st.dataframe([{"Pipeline": name, "Answerable questions": s["answerable_n"], "Hit@k": round(s["hit"], 3),
                       "All-evidence coverage": round(s["all_evidence"], 3), "MRR@k": round(s["reciprocal_rank"], 3),
                       "warm p50 (ms)": round(s["warm_p50_ms"], 1)} for name, s in data["summary"].items()], hide_index=True, use_container_width=True)
        st.caption("A hand-authored, single-report evaluation with automatic evidence matching. These are not generated-answer accuracy or production results. HyDE has not been tested live.")
        with st.expander("Scoring methodology and failure cases"):
            st.markdown((settings.root / "reports/retrieval-test.md").read_text())
    else:
        st.info("Run finrag evaluate to generate measured results.")

with about_tab:
    st.markdown("""
**Indexing:** PDF → page extraction → preserve headers and units → chunks → E5 passage embeddings → local index.

**Retrieval:** full question → E5 query embedding → search → bounded context → excerpts with page references.

**Optional generation:** evidence → DeepInfra → JSON and citation checks. HyDE drafts only help retrieve real documents.

- Similarity is not the probability of a correct answer. Hybrid retrieval does not improve every question.
- Local mode makes no automatic answerability decision. Verify the original source.
- Table handling targets this report layout. Scans need OCR; complex tables still need manual review.
- This is a reproducible single-user prototype. Tenant isolation, authentication, incremental updates, and large-scale indexing are future work.
""")
