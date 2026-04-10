"""Recommendation evaluation module for CI RAG pipeline.

Provides:
    - EvaluationReport: dataclass summarizing recommendation quality.
    - evaluate_recommendations: Aggregate statistics and consistency checks.
    - format_report: Format report as readable markdown.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationReport:
    """Report summarizing the quality and consistency of recommendations."""
    total: int = 0
    keep_count: int = 0
    relax_count: int = 0
    tighten_count: int = 0
    avg_confidence: float = 0.0
    consistency_score: float = 0.0


def evaluate_recommendations(
    recommendations: list,  # list[CIRecommendation]
) -> EvaluationReport:
    """
    Evaluate the quality and consistency of a set of recommendations.

    Args:
        recommendations: List of CIRecommendation objects.

    Returns:
        EvaluationReport object.
    """
    if not recommendations:
        return EvaluationReport()

    total = len(recommendations)
    keep_count = sum(1 for r in recommendations if r.label == "KEEP")
    relax_count = sum(1 for r in recommendations if r.label == "RELAX")
    tighten_count = sum(1 for r in recommendations if r.label == "TIGHTEN")

    # Average confidence
    avg_confidence = np.mean([r.confidence for r in recommendations])

    # Consistency score: 1.0 - std of confidence (higher = more consistent)
    confidence_std = np.std([r.confidence for r in recommendations])
    consistency_score = max(0.0, 1.0 - confidence_std)

    return EvaluationReport(
        total=total,
        keep_count=keep_count,
        relax_count=relax_count,
        tighten_count=tighten_count,
        avg_confidence=float(avg_confidence),
        consistency_score=float(consistency_score),
    )


def format_report(report: EvaluationReport) -> str:
    """
    Format an EvaluationReport as readable markdown.

    Args:
        report: EvaluationReport object.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        "## CI RAG Evaluation Report",
        "",
        f"**Total Criteria Evaluated:** {report.total}",
        "",
        "### Recommendation Breakdown",
        f"- **KEEP:** {report.keep_count} ({100*report.keep_count/report.total:.1f}%)" if report.total > 0 else "- **KEEP:** 0",
        f"- **RELAX:** {report.relax_count} ({100*report.relax_count/report.total:.1f}%)" if report.total > 0 else "- **RELAX:** 0",
        f"- **TIGHTEN:** {report.tighten_count} ({100*report.tighten_count/report.total:.1f}%)" if report.total > 0 else "- **TIGHTEN:** 0",
        "",
        "### Quality Metrics",
        f"- **Average Confidence:** {report.avg_confidence:.2f} / 1.00",
        f"- **Consistency Score:** {report.consistency_score:.2f} / 1.00",
    ]
    return "\n".join(lines)
