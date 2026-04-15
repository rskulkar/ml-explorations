"""Job opportunity comparison and ranking."""

import json
import os
import re
from typing import Optional


def run_prompt3(jobs_with_analyses: list[dict], api_key: Optional[str] = None) -> dict:
    """Analyze and rank multiple job opportunities.

    Args:
        jobs_with_analyses: List of job dicts with gap_analysis and other fields
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        Dictionary with ranked_jobs, pattern_insights, recommended_next_steps

    Raises:
        ValueError: If jobs_with_analyses is empty or JSON parsing fails
    """
    if not jobs_with_analyses:
        raise ValueError("jobs_with_analyses cannot be empty")

    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not provided")

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Take max 30 jobs, sorted by created_at descending
        jobs_to_analyze = sorted(
            jobs_with_analyses,
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )[:30]

        # Build summaries for each job
        summaries = []
        for idx, job in enumerate(jobs_to_analyze, 1):
            company = job.get("company", "Unknown")
            title = job.get("title", "Unknown")

            # Parse gap_analysis if it's a string
            gap_analysis = job.get("gap_analysis", {})
            if isinstance(gap_analysis, str):
                try:
                    gap_analysis = json.loads(gap_analysis)
                except (json.JSONDecodeError, TypeError):
                    gap_analysis = {}

            strengths = gap_analysis.get("strengths", [])[:2]
            gaps = gap_analysis.get("gaps", [])[:2]

            strengths_str = "; ".join(strengths) if strengths else "Not specified"
            gaps_str = "; ".join(gaps) if gaps else "Not specified"

            summary = f"{idx}. {company} - {title}: Strengths: {strengths_str}. Gaps: {gaps_str}."
            summaries.append(summary)

        summaries_str = "\n".join(summaries)

        system_prompt = "You are a career strategy advisor. Analyze multiple job opportunities and provide ranked recommendations. Return structured JSON only, no markdown fences."

        user_message = f"""Analyze these job opportunities and return JSON with exactly these keys:
- ranked_jobs: list of dicts each with: job_id, company, title, rank (int), fit_score (0-100), rationale (1-2 sentences)
- pattern_insights: list of 4-6 strategic observations about the candidate's positioning across all opportunities
- recommended_next_steps: list of 3-5 concrete actions the candidate should take

Opportunities:
{summaries_str}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text.strip()

        # Strip markdown fences if present
        response_text = re.sub(r"^```json\n?", "", response_text)
        response_text = re.sub(r"\n?```$", "", response_text)
        response_text = response_text.strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}. Raw response: {response_text}")
    except Exception as e:
        raise ValueError(f"Error running prompt3: {e}")
