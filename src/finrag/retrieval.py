"""Dense and BM25 retrieval, reciprocal rank fusion, and bounded evidence packs."""
from dataclasses import dataclass
import re

import numpy as np
from rank_bm25 import BM25Okapi


def lexical_tokens(text):
    english = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese + [a + b for a, b in zip(chinese, chinese[1:])]


@dataclass
class Hit:
    document: object
    score: float
    cosine: float
    score_type: str


class Retriever:
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.tokens = [lexical_tokens(d.page_content) for d in index.documents]
        self.bm25 = BM25Okapi(self.tokens)

    def retrieve(self, question, k=5, strategy="dense", doc_id=None, hypothetical=None):
        if not question.strip():
            raise ValueError("Enter a question.")
        if not 1 <= k <= 20:
            raise ValueError("top-k must be between 1 and 20.")
        if strategy not in {"dense", "hybrid", "hyde"}:
            raise ValueError("Unknown retrieval strategy")
        if strategy == "hyde" and not hypothetical:
            raise ValueError("HyDE requires a hypothetical passage; ordinary retrieval is not a HyDE substitute.")
        # HyDE draft is a PASSAGE, unlike the original user's QUERY.
        vector = self.embeddings.embed_documents([hypothetical])[0] if strategy == "hyde" else self.embeddings.embed_query(question)
        dense = self.index.search_vector(vector, max(k * 4, 20), doc_id)
        if strategy != "hybrid":
            return [Hit(self.index.documents[i], score, score, "cosine") for i, score in dense[:k]]
        eligible = [i for i, d in enumerate(self.index.documents) if not doc_id or d.metadata["doc_id"] == doc_id]
        bm_scores = self.bm25.get_scores(lexical_tokens(question))
        # Zero-overlap lexical matches must not perturb dense ranking (e.g. Chinese to English).
        lexical = sorted((i for i in eligible if bm_scores[i] > 0), key=lambda i: (-bm_scores[i], i))[:max(k * 4, 20)]
        rrf = {}
        for ranking in [[i for i, _ in dense], lexical]:
            for rank, i in enumerate(ranking, 1):
                rrf[i] = rrf.get(i, 0) + 1 / (60 + rank)
        cosine_scores = self.index.vectors @ np.asarray(vector)
        return [Hit(self.index.documents[i], float(rrf[i]), float(cosine_scores[i]), "rrf")
                for i in sorted(rrf, key=lambda i: (-rrf[i], i))[:k]]


def pack_context(hits, max_bytes):
    """Keep complete chunks (including headers); never truncate a financial row.

    UTF-8 byte budget is an explicit size proxy, not a provider-token estimate.
    """
    items, seen, used = [], set(), 0
    for hit in hits:
        document = hit.document
        identity = (document.metadata["doc_id"], document.page_content)
        if identity in seen:
            continue
        citation = f"S{len(items) + 1}"
        meta = document.metadata
        header_pages = meta.get("header_pages") or ([meta["header_page"]] if meta.get("header_page") else [])
        label = f"[{citation}] {meta['title']} | PDF page {meta['page']}"
        if header_pages and header_pages != [meta["page"]]:
            label += f" | table header from PDF pages {', '.join(map(str, header_pages))}"
        if meta.get("quality_warning"):
            label += " | Warning: table layout incomplete, verify original PDF before using numeric claims"
        block = label + "\n" + document.page_content + "\n\n"
        size = len(block.encode("utf-8"))
        if used + size > max_bytes:
            continue
        seen.add(identity)
        used += size
        items.append({"citation": citation, "text": document.page_content, "metadata": meta,
                      "score": hit.score, "cosine": hit.cosine, "score_type": hit.score_type, "context": block})
    return items, used
