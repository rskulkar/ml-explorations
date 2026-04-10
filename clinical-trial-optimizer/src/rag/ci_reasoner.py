"""Competitive Intelligence reasoner module for CI RAG pipeline.

Provides:
    - CIRecommendation: dataclass representing a criterion recommendation.
    - reason_about_criterion: LLM-based reasoning on whether to KEEP/RELAX/TIGHTEN a criterion.
"""
from dataclasses import dataclass, field


@dataclass
class CIRecommendation:
    """Recommendation on whether to keep, relax, or tighten a criterion."""
    criterion_text: str
    label: str  # "KEEP" | "RELAX" | "TIGHTEN"
    rationale: str
    evidence_trials: list[str] = field(default_factory=list)  # NCT IDs
    suggested_wording: str = ""
    confidence: float = 0.0


def reason_about_criterion(
    source_criterion: str,
    retrieved_chunks: list,  # list[RetrievalResult]
    api_key: str | None = None,
) -> CIRecommendation:
    """
    Use Claude Haiku to reason about whether to KEEP, RELAX, or TIGHTEN a criterion.

    Args:
        source_criterion: The source criterion text to reason about.
        retrieved_chunks: List of RetrievalResult objects from hybrid retrieval.
        api_key: Optional Anthropic API key.

    Returns:
        CIRecommendation object.
    """
    # TODO: implement LLM-based reasoning
    pass
