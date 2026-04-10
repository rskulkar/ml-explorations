"""Retrieval evaluation module for CI RAG pipeline.

Provides:
    - RetrievalMetrics: dataclass for retrieval performance metrics.
    - evaluate_retrieval: Evaluate retrieval results against relevant trials.
"""
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    coverage: float = 0.0


def evaluate_retrieval(
    results: list,  # list[RetrievalResult]
    relevant_nct_ids: list[str],
    k: int = 10,
) -> RetrievalMetrics:
    """
    Evaluate retrieval results against a set of relevant trials.

    Args:
        results: List of RetrievalResult objects from retrieve().
        relevant_nct_ids: List of relevant NCT IDs (ground truth).
        k: Number of results to evaluate (default 10).

    Returns:
        RetrievalMetrics object.
    """
    relevant_set = set(relevant_nct_ids)
    results_to_eval = results[:k]

    # Precision@K: count relevant results in top-k
    relevant_count = 0
    for result in results_to_eval:
        if result.chunk.source_nct_id in relevant_set:
            relevant_count += 1
    precision_at_k = relevant_count / k if k > 0 else 0.0

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for rank, result in enumerate(results_to_eval, start=1):
        if result.chunk.source_nct_id in relevant_set:
            mrr = 1.0 / rank
            break

    # Coverage: unique relevant NCT IDs found / total relevant NCT IDs
    retrieved_nct_ids = {result.chunk.source_nct_id for result in results}
    found_relevant = len(retrieved_nct_ids & relevant_set)
    coverage = found_relevant / len(relevant_set) if len(relevant_set) > 0 else 0.0

    return RetrievalMetrics(
        precision_at_k=precision_at_k,
        mrr=mrr,
        coverage=coverage,
    )
