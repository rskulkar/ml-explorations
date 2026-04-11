"""Criterion extraction and chunking module for CI RAG pipeline.

Provides:
    - CriterionChunk: dataclass representing an extracted I/E criterion.
    - parse_criteria_via_llm: LLM-based I/E criterion extraction from text.
    - chunk_competing_trials: Extract and chunk criteria from competing trials table.
"""
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)


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
    system_prompt = (
        "You are a clinical trial protocol parser. Extract each individual inclusion and exclusion criterion "
        "as a separate item. Output only valid JSON, no markdown fences."
    )
    user_message = f"""Parse the following I/E criteria text into individual criteria. Return a JSON array where each element has:
   - type: 'inclusion' or 'exclusion'
   - criterion_number: integer starting from 1 within each type
   - text: exact criterion text, cleaned of numbering/bullets
   - treatment_related: true if criterion mentions medications, drug classes, prior treatment, treatment lines, or systemic therapies; false otherwise

   I/E criteria text:
   {ie_text}"""

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.DOTALL).strip()
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            parsed = [parsed]

        chunks = []
        for item in parsed:
            chunk = CriterionChunk(
                text=item.get("text", ""),
                criterion_type=item.get("type", ""),
                source_nct_id=nct_id,
                source_trial_name=trial_name,
                trial_phase=metadata.get("phase"),
                trial_status=metadata.get("status"),
                trial_sponsor=metadata.get("sponsor"),
                metadata={
                    "treatment_related": item.get("treatment_related", False),
                    "criterion_number": item.get("criterion_number"),
                    **metadata,
                },
            )
            chunks.append(chunk)
        return chunks
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse criteria JSON for {nct_id}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to extract criteria for {nct_id}: {e}")
        return []


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
    lines = competing_trials_md.strip().split("\n")

    # Skip preamble (search note lines starting with "_")
    table_start = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and "NCT ID" in line:
            table_start = i
            break

    # Skip header and separator
    rows = lines[table_start + 2 :]

    # Parse rows
    all_chunks = []
    row_count = 0
    for i, line in enumerate(rows):
        if not line.strip() or not line.startswith("|"):
            continue

        row_count += 1
        cells = [c.strip() for c in line.split("|")[1:-1]]  # Skip first/last empty cells
        if len(cells) < 9:
            continue

        nct_id = cells[0]
        trial_name = cells[1]
        phase = cells[2]
        status = cells[3]
        sponsor = cells[4]
        ie_criteria_text = cells[8]  # Last column

        # Skip rows with missing or placeholder I/E criteria
        if ie_criteria_text in ("—", "", "N/A"):
            continue

        print(f"Chunking trial {row_count}: {nct_id}")
        metadata = {
            "phase": phase,
            "status": status,
            "sponsor": sponsor,
        }

        chunks = parse_criteria_via_llm(ie_criteria_text, nct_id, trial_name, metadata, api_key)
        all_chunks.extend(chunks)

    return all_chunks
