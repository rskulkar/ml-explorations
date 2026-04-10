"""Hybrid retrieval module (FAISS + BM25 with RRF) for CI RAG pipeline.

Provides:
    - RetrievalResult: dataclass representing a ranked retrieved chunk.
    - retrieve: Hybrid retrieval using Reciprocal Rank Fusion (RRF).
"""
from dataclasses import dataclass
from typing import Optional


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
    # TODO: implement hybrid retrieval with RRF
    pass
