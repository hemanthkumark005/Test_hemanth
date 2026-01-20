# tfidf_pdf_validation_poc.py
# pip install pymupdf scikit-learn numpy

import re
import math
import fitz
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 0) HELPERS
# -----------------------------
def normalize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def approx_tokens(text: str) -> int:
    # rough: 1 token ~= 0.75 words
    w = len(text.split())
    return int(math.ceil(w / 0.75))

# -----------------------------
# 1) PDF TEXT EXTRACTION
# -----------------------------
def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append({"page": i + 1, "text": normalize_text(text)})
    return pages

# -----------------------------
# 2) CHUNK PDF1 (SOURCE)
# -----------------------------
def chunk_pdf_by_words(pages: List[Dict], pdf_name: str, target_words: int = 420, overlap_words: int = 80) -> List[Dict]:
    chunks = []
    buf = []
    start_page = None
    end_page = None
    cid = 1

    def flush():
        nonlocal cid, buf, start_page, end_page
        if not buf:
            return
        chunks.append({
            "chunk_id": f"{pdf_name}_C{cid:04d}",
            "page_start": start_page,
            "page_end": end_page,
            "text": " ".join(buf).strip()
        })
        cid += 1

    for pg in pages:
        words = pg["text"].split()
        if not words:
            continue
        for w in words:
            if not buf:
                start_page = pg["page"]
            buf.append(w)
            end_page = pg["page"]

            if len(buf) >= target_words:
                flush()
                buf = buf[-overlap_words:] if overlap_words > 0 else []

    flush()
    return chunks

# -----------------------------
# 3) EXTRACT "REQUIRED ONLY" FROM PDF2
# -----------------------------
def extract_pdf2_checks_by_headings(pages: List[Dict], section_titles: List[str]) -> List[Dict]:
    """
    Controlled extraction:
    - You provide section_titles you care about (e.g., ["Scope", "Definitions", "Exclusions"])
    - We capture text after a heading until the next heading
    Works well when PDF2 is structured.
    """
    titles = [t.lower().strip() for t in section_titles]
    checks = []

    current_title = None
    current_lines = []
    current_page = None

    def flush():
        nonlocal current_title, current_lines, current_page
        if current_title and current_lines:
            checks.append({
                "check_id": f"CHK_{len(checks)+1:03d}",
                "title": current_title,
                "page": current_page,
                "text": " ".join(current_lines).strip()
            })
        current_title = None
        current_lines = []
        current_page = None

    for pg in pages:
        lines = [ln.strip() for ln in pg["text"].splitlines() if ln.strip()]
        for ln in lines:
            ln_low = ln.lower()

            # heading match (exact or prefix)
            if any(ln_low == t or ln_low.startswith(t + " ") for t in titles):
                flush()
                current_title = ln
                current_page = pg["page"]
                continue

            if current_title:
                current_lines.append(ln)

    flush()
    return checks

def extract_pdf2_checks_by_bullets(pages: List[Dict], min_len: int = 25) -> List[Dict]:
    """
    Alternative extraction:
    - Pull only bullet points from PDF2 (if that's what matters)
    """
    bullet_re = re.compile(r"^(\u2022|\-|\*|\d+\)|\(\d+\)|[a-zA-Z]\))\s+")
    checks = []
    for pg in pages:
        for ln in pg["text"].splitlines():
            ln = ln.strip()
            if bullet_re.match(ln) and len(ln) >= min_len:
                checks.append({
                    "check_id": f"CHK_{len(checks)+1:03d}",
                    "title": "BULLET",
                    "page": pg["page"],
                    "text": ln
                })
    return checks

# -----------------------------
# 4) BUILD TF-IDF INDEX (IN MEMORY)
# -----------------------------
def build_tfidf(pdf1_chunks: List[Dict]) -> Dict:
    texts = [c["text"] for c in pdf1_chunks]
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200_000
    )
    M = vec.fit_transform(texts)
    return {"vectorizer": vec, "matrix": M}

def tfidf_encode(vec: TfidfVectorizer, texts: List[str]):
    return vec.transform(texts)

def cosine_top_k(query_vec, doc_matrix, k: int = 6) -> List[int]:
    sims = cosine_similarity(query_vec, doc_matrix).flatten()
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return idx.tolist(), sims

# -----------------------------
# 5) EVIDENCE COMPRESSION (OPTIONAL BUT IMPORTANT FOR 2K CONTEXT)
# -----------------------------
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def compress_evidence_extractive(query_text: str, hit_chunks: List[Dict], max_tokens: int = 650) -> str:
    """
    Extract sentences most similar to query_text (TF-IDF on sentence level).
    """
    sentences = []
    meta = []

    for c in hit_chunks:
        sents = [s.strip() for s in SENT_SPLIT_RE.split(c["text"]) if len(s.strip()) > 30]
        for s in sents:
            sentences.append(s)
            meta.append(c)

    if not sentences:
        return ""

    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=80_000)
    X = vec.fit_transform(sentences + [query_text])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    order = np.argsort(-sims)

    packed = []
    used = 0
    per_chunk = {}

    for i in order:
        if sims[i] <= 0:
            break
        c = meta[i]
        cid = c["chunk_id"]
        per_chunk[cid] = per_chunk.get(cid, 0) + 1
        if per_chunk[cid] > 3:
            continue

        line = f"[{cid} | p{c['page_start']}-{c['page_end']}] {sentences[i]}"
        t = approx_tokens(line)
        if used + t > max_tokens:
            continue

        packed.append(line)
        used += t
        if used >= max_tokens:
            break

    return "\n".join(packed)

# -----------------------------
# 6) PIPELINE: MATCH PDF2 CHECKS -> PDF1 EVIDENCE
# -----------------------------
def match_checks_to_pdf1(pdf1_chunks: List[Dict], tfidf_index: Dict, checks: List[Dict], top_k: int = 6) -> List[Dict]:
    vec = tfidf_index["vectorizer"]
    M = tfidf_index["matrix"]

    results = []
    for chk in checks:
        qv = tfidf_encode(vec, [chk["text"]])
        idxs, sims = cosine_top_k(qv, M, k=top_k)
        hit_chunks = [pdf1_chunks[i] for i in idxs if sims[i] > 0]

        evidence = compress_evidence_extractive(chk["text"], hit_chunks, max_tokens=650)

        results.append({
            "check_id": chk["check_id"],
            "pdf2_page": chk["page"],
            "title": chk["title"],
            "check_text": chk["text"],
            "top_matches": [
                {
                    "chunk_id": c["chunk_id"],
                    "pages": f"p{c['page_start']}-{c['page_end']}",
                    "similarity": float(sims[pdf1_chunks.index(c)]) if c in pdf1_chunks else None
                }
                for c in hit_chunks[:top_k]
            ],
            "evidence": evidence
        })
    return results

# -----------------------------
# 7) DRIVER
# -----------------------------
def run_poc(pdf1_path: str, pdf2_path: str, mode: str = "headings", section_titles: List[str] = None, top_k: int = 6) -> List[Dict]:
    # PDF1
    pdf1_pages = extract_pdf_pages(pdf1_path)
    pdf1_chunks = chunk_pdf_by_words(pdf1_pages, pdf_name="PDF1", target_words=420, overlap_words=80)
    tfidf_index = build_tfidf(pdf1_chunks)

    # PDF2 (required only)
    pdf2_pages = extract_pdf_pages(pdf2_path)
    if mode == "headings":
        if not section_titles:
            raise ValueError("For mode='headings', provide section_titles list.")
        checks = extract_pdf2_checks_by_headings(pdf2_pages, section_titles=section_titles)
    elif mode == "bullets":
        checks = extract_pdf2_checks_by_bullets(pdf2_pages)
    else:
        raise ValueError("mode must be 'headings' or 'bullets'")

    # match
    results = match_checks_to_pdf1(pdf1_chunks, tfidf_index, checks, top_k=top_k)
    return results


if __name__ == "__main__":
    PDF1 = "PDF1.pdf"
    PDF2 = "PDF2.pdf"

    # Example: validate only these PDF2 sections
    TITLES = ["Scope", "Definitions", "Exclusions", "Eligibility", "Responsibilities"]

    out = run_poc(PDF1, PDF2, mode="headings", section_titles=TITLES, top_k=6)

    # Print first 2 checks
    for r in out[:2]:
        print("=" * 90)
        print("CHECK:", r["check_id"], "| PDF2 page:", r["pdf2_page"], "| Title:", r["title"])
        print("CHECK TEXT (preview):", r["check_text"][:250])
        print("EVIDENCE (preview):\n", r["evidence"][:900])
