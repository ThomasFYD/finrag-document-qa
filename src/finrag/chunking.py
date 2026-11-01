"""A fair fixed-size baseline and a layout-aware semantic alternative.

Tables use repeated source headers; prose groups adjacent semantic units.
This rule-based parser supports this report family, not arbitrary PDF tables.
"""
from dataclasses import dataclass
import hashlib
import re

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

VERSION = "layout-semantic-v1"


@dataclass
class Block:
    text: str
    section: str
    kind: str = "prose"
    header: str = ""
    header_page: int | None = None
    header_pages: tuple[int, ...] = ()


def compact(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_numeric_row(line: str) -> bool:
    # A label followed by aligned numeric columns, not a wrapped paragraph.
    columns = re.split(r"\s{2,}", line.strip())
    return (len(columns) >= 2 and bool(re.match(r"[$(\d-]", columns[1]))
            and bool(re.search(r"\d", " ".join(columns[1:])))
            and bool(re.search(r"[A-Za-z]", columns[0])))


def page_blocks(page: Document, carry: dict) -> list[Block]:
    lines = page.page_content.splitlines()
    blocks, buffer = [], []
    section = carry.get("section", "")
    header = carry.get("header", "")
    header_page = carry.get("header_page")
    header_pages = carry.get("header_pages", ())
    table_active = bool(header)

    # Some report tables put their date columns at the top of the following page.
    # Extend an inherited header only with actual leading column-header lines.
    if table_active:
        leading = []
        for line in lines:
            s = line.strip()
            if (not s or re.match(r"(?:Three Months|Six Months|July |April |January |20\d\d)", s)):
                if s:
                    leading.append(s)
            else:
                break
        if leading:
            header += "\n" + "\n".join(leading)
            header_pages = tuple(sorted(set(header_pages) | {page.metadata["page"]}))
            lines = lines[len(leading):]

    def flush():
        if buffer:
            text = " ".join(buffer).strip()
            # Sentence units let the semantic boundary actually affect grouping.
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z“])", text):
                if sentence:
                    blocks.append(Block(sentence, section))
            buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if "CONDENSED CONSOLIDATED" in s or s in {"GAAP", "Non-GAAP"} or "RECONCILIATION OF GAAP TO NON-GAAP" in s:
            flush()
            section = compact(s)
            h = [section]
            j = i + 1
            while j < len(lines) and not is_numeric_row(lines[j]):
                if lines[j].strip():
                    h.append(compact(lines[j]))
                j += 1
            # Summary header contains digits (FY25); collect it explicitly.
            if s in {"GAAP", "Non-GAAP"}:
                h, j = [s], i + 1
                while j < len(lines) and not lines[j].strip().startswith("Revenue"):
                    if lines[j].strip():
                        h.append(compact(lines[j]))
                    j += 1
            header, header_page = "\n".join(h), page.metadata["page"]
            header_pages = (header_page,)
            table_active = True
            i = j
            continue
        if table_active and is_numeric_row(line):
            flush()
            blocks.append(Block(compact(s), section, "table", header, header_page, header_pages))
        elif table_active and s in {"About NVIDIA", "Outlook", "Highlights", "CFO Commentary", "Conference Call and Webcast Information", "Non-GAAP Measures"}:
            flush()
            table_active, header, section = False, "", s
        elif table_active and re.fullmatch(r"[\s\d,$().%+-]+", s):
            flush()
            blocks.append(Block(compact(s), section, "table", header, header_page, header_pages))
        elif table_active and (not s.endswith(".") and len(s) < 100):
            # Account subheadings are source context, not answers or invented labels.
            if s != "NVIDIA CORPORATION":
                blocks.append(Block(compact(s), section, "table", header, header_page, header_pages))
        else:
            if table_active:
                table_active = False
                header = ""
            # Short title lines delimit sections; bullets remain prose.
            if len(s) < 70 and not re.search(r"[.!?:]$", s) and not re.search(r"\d", s) and len(s.split()) <= 9:
                flush()
                section = s
            else:
                if line.startswith("      "):
                    flush()
                buffer.append(compact(s))
        i += 1
    flush()
    carry.update(section=section, header=header if table_active else "", header_page=header_page, header_pages=header_pages)
    return blocks


def make_chunk(page: Document, text: str, method: str, ordinal: int, **metadata) -> Document:
    digest = hashlib.sha256((page.metadata["source_sha256"] + str(page.metadata["page"]) + method + str(ordinal) + text).encode()).hexdigest()[:20]
    return Document(page_content=text, metadata={**page.metadata, **metadata, "chunk_id": digest, "method": method})


def split_pages(pages, embeddings, settings, method: str) -> list[Document]:
    if method not in {"baseline", "semantic"}:
        raise ValueError("Unknown chunk method")
    chunks = []
    carry = {}
    previous_doc = None
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.overlap_tokens, length_function=embeddings.token_count,
        separators=["\n\n", "\n", ". ", " ", ""])
    for page in pages:
        if previous_doc != page.metadata["doc_id"]:
            carry = {}
        previous_doc = page.metadata["doc_id"]
        prefix = f"{page.metadata['title']}\n"
        if method == "baseline":
            for text in splitter.split_text(page.page_content):
                chunks.append(make_chunk(page, prefix + text, method, len(chunks), kind="fixed", section=""))
            continue
        blocks = page_blocks(page, carry)
        # Keep each unit within the same budget, including heading and repeated table header.
        units = []
        for b in blocks:
            context = prefix + (b.header if b.kind == "table" else b.section) + "\n"
            room = settings.chunk_tokens - embeddings.token_count(context) - 8
            if room < 40:
                raise ValueError(f"Table header on page {page.metadata['page']} is too long; manual review is required.")
            small_splitter = RecursiveCharacterTextSplitter(chunk_size=room, chunk_overlap=0,
                length_function=embeddings.token_count, separators=[". ", " ", ""])
            for text in small_splitter.split_text(b.text):
                units.append(Block(text, b.section, b.kind, b.header, b.header_page, b.header_pages))
        prose = [u.text for u in units if u.kind == "prose"]
        prose_vectors = iter(embeddings.embed_documents(prose))
        vectors = [next(prose_vectors) if u.kind == "prose" else None for u in units]
        current, previous_vector, previous = [], None, None

        def emit():
            if current:
                b = current[0]
                ctx = b.header if b.kind == "table" else b.section
                text = prefix + ctx + "\n" + "\n".join(x.text for x in current)
                metadata = {"kind": b.kind, "section": b.section}
                if b.kind == "table":
                    metadata.update(header_page=b.header_page, header_pages=list(b.header_pages), table_header=b.header)
                elif "Three Months Ended" in text or "Six Months Ended" in text:
                    metadata["quality_warning"] = "Complex table layout is incomplete. Units or column relationships may be missing; verify the original PDF."
                chunks.append(make_chunk(page, text, method, len(chunks), **metadata))

        for unit, vector in zip(units, vectors):
            ctx = unit.header if unit.kind == "table" else unit.section
            candidate = prefix + ctx + "\n" + "\n".join([x.text for x in current] + [unit.text])
            boundary = previous and (unit.kind != previous.kind or ctx != (previous.header if previous.kind == "table" else previous.section))
            # Neighboring sentence embeddings decide semantic boundaries in prose.
            semantic_break = vector is not None and previous_vector is not None and 1 - float(np.dot(vector, previous_vector)) > settings.semantic_distance
            if current and (boundary or semantic_break or embeddings.token_count(candidate) > settings.chunk_tokens):
                emit(); current = []
            current.append(unit)
            previous, previous_vector = unit, vector
        emit()
    if not chunks:
        raise ValueError("No indexable text was extracted.")
    for chunk in chunks:
        chunk.metadata["tokens"] = embeddings.token_count(chunk.page_content)
        if chunk.metadata["tokens"] + 6 > 512:
            raise ValueError("A chunk exceeds the embedding model length limit.")
    return chunks
