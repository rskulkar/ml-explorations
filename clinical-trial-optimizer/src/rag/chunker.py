"""Criterion extraction and chunking module for CI RAG pipeline.

Provides:
    - CriterionChunk: dataclass representing an extracted I/E criterion.
    - parse_criteria_via_llm: LLM-based I/E criterion extraction from text.
    - chunk_competing_trials: Extract and chunk criteria from competing trials table.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CriterionChunk:
    """Represents an extracted clinical trial criterion."""
    text: str
    criterion_type: str  # "inclusion" | "exclusion" | "suggested_addition"
    source_nct_id: Optional[str] = None
    source_trial_name: Optional[str] = None
    trial_phase: Optional[str] = None
    trial_status: Optional[str] = None
    trial_sponsor: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None


def parse_criteria_via_llm(
    ie_text: str,
    nct_id: str,
    trial_name: str,
    metadata: dict,
    api_key: str | None = None,
) -> list[CriterionChunk]:
    """
    Use Claude Haiku to parse I/E criteria text into discrete criterion chunks.

    Args:
        ie_text: Raw I/E criteria text from a trial.
        nct_id: ClinicalTrials.gov NCT ID.
        trial_name: Trial brief title.
        metadata: Trial metadata (phase, status, sponsor, etc.).
        api_key: Optional Anthropic API key.

    Returns:
        List of CriterionChunk objects.
    """
    # TODO: implement LLM-based criterion extraction
    pass


def chunk_competing_trials(
    competing_trials_md: str,
    api_key: str | None = None,
) -> list[CriterionChunk]:
    """
    Extract and chunk criteria from a markdown table of competing trials.

    Args:
        competing_trials_md: Markdown table of competing trials (from fetch_competing_trials).
        api_key: Optional Anthropic API key.

    Returns:
        List of CriterionChunk objects extracted from all trials in the table.
    """
    # TODO: implement competing trials chunking
    pass
