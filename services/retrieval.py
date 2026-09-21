"""
retrieval.py — DocuMind v2
Replaces naive FAISS IndexFlatL2 with:
  1. BM25 sparse retrieval  (rank_bm25)
  2. Dense FAISS retrieval  (all-MiniLM-L6-v2, unchanged embedding model)
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. FlashRank cross-encoder reranker  (no API key needed, runs locally)

Drop-in replacement: retrieve_chunks() signature unchanged.
New:   retrieve_hybrid()  — used by chat and eval routers.
"""

import json
import os
import math
import logging
from typing import List, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi          # pip install rank-bm25
from flashrank import Ranker, RerankRequest  # pip install flashrank

from config import INDEX_DIR
from services.embeddings import get_model

logger = logging.getLogger(__name__)

# ── FlashRank reranker (ms-marco-MiniLM-L-12-v2, ~120 MB, cached after first load)
_reranker: Ranker | None = None

def get_reranker() -> Ranker:
    global _reranker
    if _reranker is None:
        _reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
    return _reranker


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _load_index_and_meta(doc_id: str):
    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    meta_path  = os.path.join(INDEX_DIR, f"{doc_id}_meta.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found for document {doc_id}")
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return index, meta


def _dense_retrieve(index, meta, query: str, top_k: int) -> List[Dict]:
    """Standard FAISS L2 search → returns list of {chunk, chunk_id, page, score}."""
    chunks = meta["chunks"]
    pages  = meta.get("pages", [1] * len(chunks))
    qvec   = get_model().encode([query]).astype(np.float32)
    distances, indices = index.search(qvec, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk":    chunks[idx],
                "chunk_id": int(idx),
                "page":     pages[idx],
                "score":    float(distances[0][rank]),
                "dense_rank": rank,
            })
    return results


def _bm25_retrieve(meta, query: str, top_k: int) -> List[Dict]:
    """BM25 Okapi retrieval over tokenised chunk corpus."""
    chunks = meta["chunks"]
    pages  = meta.get("pages", [1] * len(chunks))
    # Simple whitespace tokenisation (good enough; you can swap in spaCy)
    tokenised_corpus = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised_corpus)
    tokenised_query  = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "chunk":    chunks[idx],
            "chunk_id": int(idx),
            "page":     pages[idx],
            "score":    float(scores[idx]),
            "bm25_rank": rank,
        })
    return results


def _rrf_fuse(
    dense_results: List[Dict],
    bm25_results:  List[Dict],
    k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight:  float = 0.4,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion.
    score(d) = Σ  weight / (k + rank(d))
    k=60 is the standard RRF constant (prevents top-rank dominance).
    """
    rrf_scores: Dict[int, float] = {}
    chunk_map:  Dict[int, Dict]  = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + dense_weight / (k + rank + 1)
        chunk_map[cid]  = item

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight  / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = item

    fused = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = rrf_score
        fused.append(entry)

    return fused


def _rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """
    FlashRank cross-encoder reranker.
    Takes the RRF-fused candidates and re-scores them with a cross-encoder.
    """
    ranker    = get_reranker()
    passages  = [{"id": c["chunk_id"], "text": c["chunk"]} for c in candidates]
    request   = RerankRequest(query=query, passages=passages)
    ranked    = ranker.rerank(request)

    # ranked is a list of {"id": ..., "text": ..., "score": ...}
    id_to_score = {r["id"]: r["score"] for r in ranked}
    for item in candidates:
        item["rerank_score"] = id_to_score.get(item["chunk_id"], 0.0)

    reranked = sorted(candidates, key=lambda x: -x["rerank_score"])
    return reranked[:top_k]


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def retrieve_chunks(doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    LEGACY ENTRY POINT — kept for backward compatibility.
    Now internally runs hybrid retrieval + reranking.
    """
    return retrieve_hybrid(doc_id, query, top_k=top_k)


def retrieve_hybrid(
    doc_id: str,
    query:  str,
    top_k:  int  = 5,
    overretrieve_factor: int  = 4,   # fetch 4× then rerank down to top_k
    use_reranker: bool = True,
) -> List[Dict]:
    """
    Full pipeline:
        Dense (FAISS)  +  Sparse (BM25)  →  RRF fusion  →  FlashRank rerank

    Args:
        overretrieve_factor: multiplier for initial candidates before reranking
    Returns:
        List of top_k dicts with keys: chunk, chunk_id, page, score, rrf_score, rerank_score
    """
    index, meta = _load_index_and_meta(doc_id)
    candidate_k  = top_k * overretrieve_factor

    dense_results = _dense_retrieve(index, meta, query, candidate_k)
    bm25_results  = _bm25_retrieve(meta, query, candidate_k)
    fused         = _rrf_fuse(dense_results, bm25_results)

    if use_reranker and len(fused) > 0:
        final = _rerank(query, fused, top_k)
    else:
        final = fused[:top_k]

    # Normalise key names so downstream code (chat router, eval) stays unchanged
    for item in final:
        item.setdefault("score", item.get("rerank_score", item.get("rrf_score", 0.0)))

    logger.info(
        "hybrid_retrieve doc=%s candidates=%d dense=%d bm25=%d final=%d",
        doc_id, len(fused), len(dense_results), len(bm25_results), len(final),
    )
    return final
