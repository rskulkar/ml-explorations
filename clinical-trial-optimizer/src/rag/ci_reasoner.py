"""Competitive Intelligence reasoner module for CI RAG pipeline.

Provides:
    - CIRecommendation: dataclass representing a criterion recommendation.
    - reason_about_criterion: LLM-based reasoning on whether to KEEP/RELAX/TIGHTEN a criterion.
"""
import json
import re
import logging
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)


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
    Use Claude Sonnet to reason about whether to KEEP, RELAX, or TIGHTEN a criterion.

    Args:
        source_criterion: The source criterion text to reason about.
        retrieved_chunks: List of RetrievalResult objects from hybrid retrieval.
        api_key: Optional Anthropic API key.

    Returns:
        CIRecommendation object.
    """
    # Build context from retrieved chunks
    context_lines = []
    for i, result in enumerate(retrieved_chunks, start=1):
        nct_id = result.chunk.source_nct_id or "UNKNOWN"
        trial_name = result.chunk.source_trial_name or "Unknown Trial"
        criterion_type = result.chunk.criterion_type
        text = result.chunk.text
        rrf_score = result.rrf_score

        context_lines.append(
            f"{i}. [{nct_id}] {trial_name}\n"
            f"   Type: {criterion_type}\n"
            f"   Criterion: {text}\n"
            f"   Relevance score: {rrf_score:.3f}"
        )

    context = "\n".join(context_lines)

    system_prompt = (
        "You are a clinical trial eligibility criteria expert. Analyze a source criterion against evidence "
        "from competing trials and make a structured recommendation. Output only valid JSON."
    )

    user_message = f"""Source criterion to evaluate:
{source_criterion}

Evidence from {len(retrieved_chunks)} competing trials:
{context}

Based on this competitive evidence, evaluate the source criterion and output JSON with exactly these keys:
- label: one of KEEP, RELAX, or TIGHTEN
- rationale: 2-3 sentence clinical justification referencing specific competing trials
- evidence_trials: list of NCT IDs that most influenced this recommendation
- suggested_wording: if RELAX or TIGHTEN, provide revised criterion text; if KEEP, repeat original
- confidence: float 0.0-1.0 based on evidence quality and consistency"""

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.DOTALL).strip()
        parsed = json.loads(raw)

        return CIRecommendation(
            criterion_text=source_criterion,
            label=parsed.get("label", "KEEP"),
            rationale=parsed.get("rationale", ""),
            evidence_trials=parsed.get("evidence_trials", []),
            suggested_wording=parsed.get("suggested_wording", source_criterion),
            confidence=float(parsed.get("confidence", 0.0)),
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reasoning JSON: {e}")
        return CIRecommendation(
            criterion_text=source_criterion,
            label="KEEP",
            rationale="Parse error",
            confidence=0.0,
        )
    except Exception as e:
        logger.warning(f"Failed to reason about criterion: {e}")
        return CIRecommendation(
            criterion_text=source_criterion,
            label="KEEP",
            rationale="Reasoning error",
            confidence=0.0,
        )
