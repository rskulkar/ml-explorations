"""Hybrid retrieval module (FAISS + BM25 with RRF) for CI RAG pipeline.

Provides:
    - RetrievalResult: dataclass representing a ranked retrieved chunk.
    - retrieve: Hybrid retrieval using Reciprocal Rank Fusion (RRF).
"""
import logging
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Ranked retrieval result from hybrid search."""
    chunk: object  # CriterionChunk
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0


def retrieve(
    query_text: str,
    index,  # HybridIndex
    model,  # SentenceTransformer
    k: int = 10,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    Perform hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Args:
        query_text: Query criterion text.
        index: HybridIndex object.
        model: SentenceTransformer for encoding the query.
        k: Number of results to return.
        rrf_k: RRF parameter (default 60).

    Returns:
        List of RetrievalResult objects, ranked by RRF score.
    """
    # Encode query and normalise for cosine similarity
    query_embedding = model.encode([query_text])
    faiss.normalize_L2(query_embedding)

    # Dense search: FAISS with inner product on normalised vectors
    distances, indices = index.faiss_index.search(query_embedding.astype(np.float32), k)
    dense_results = {idx: (rank + 1) for rank, idx in enumerate(indices[0])}

    # Sparse search: BM25
    query_tokens = query_text.lower().split()
    bm25_scores = index.bm25.get_scores(query_tokens)
    # Get top-k by BM25 score
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    sparse_results = {idx: (rank + 1) for rank, idx in enumerate(top_k_indices)}

    # RRF fusion: combine dense and sparse rankings
    # RRF score = sum of 1/(rrf_k + rank) for each signal
    rrf_scores = {}
    for idx in set(dense_results.keys()) | set(sparse_results.keys()):
        score = 0.0
        if idx in dense_results:
            score += 1.0 / (rrf_k + dense_results[idx])
        if idx in sparse_results:
            score += 1.0 / (rrf_k + sparse_results[idx])
        rrf_scores[idx] = score

    # Sort by RRF score and return top-k
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (idx, rrf_score) in enumerate(sorted_indices):
        result = RetrievalResult(
            chunk=index.chunks[idx],
            dense_rank=dense_results.get(idx),
            sparse_rank=sparse_results.get(idx),
            rrf_score=rrf_score,
        )
        results.append(result)

    return results
