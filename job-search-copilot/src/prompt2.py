"""Interview preparation materials."""

import json
import os
import re
from typing import Optional


def run_prompt2(
    gap_analysis: dict,
    tailored_resume: str,
    company_profile: dict,
    interviewer_context: str,
    system_prompt_override: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Generate interview preparation materials.

    Args:
        gap_analysis: Gap analysis from prompt1
        tailored_resume: Tailored resume text
        company_profile: Company profile dict
        interviewer_context: Interviewer context string
        system_prompt_override: Optional custom system prompt
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        Dictionary with behavioural_star, technical_questions, follow_up_probes, tone_notes

    Raises:
        ValueError: If JSON parsing fails
    """
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not provided")

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = "You are an expert interview coach. Prepare targeted interview preparation based on the job analysis and interviewer profile. Return structured JSON only, no markdown fences."

        gap_analysis_str = json.dumps(gap_analysis) if isinstance(gap_analysis, dict) else str(gap_analysis)
        company_profile_str = json.dumps(company_profile) if isinstance(company_profile, dict) else str(company_profile)

        user_message = f"""Prepare interview materials for this candidate and return JSON with exactly these keys:
- behavioural_star: list of 5 dicts, each with keys: question, situation, task, action, result
- technical_questions: list of 5 dicts, each with keys: question, ideal_answer
- follow_up_probes: list of 5 smart questions the candidate should ask the interviewer
- tone_notes: string with communication style recommendations for this specific interviewer

Gap Analysis: {gap_analysis_str}
Tailored Resume: {tailored_resume}
Company Profile: {company_profile_str}
Interviewer Context: {interviewer_context}"""

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
        raise ValueError(f"Error running prompt2: {e}")
