"""Resume tailoring and job analysis."""

import json
import os
import re
from typing import Optional


def run_prompt1(
    resume_text: str,
    jd_text: str,
    company: str,
    api_key: Optional[str] = None,
) -> dict:
    """Analyze job application with resume and JD.

    Args:
        resume_text: Resume text content
        jd_text: Job description text
        company: Target company name
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        Dictionary with tailored_resume, strengths, gaps, similar_companies, similar_roles, live_openings_queries

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

        system_prompt = "You are an expert career coach and resume writer. Analyze the job description and resume, then return structured JSON only, no markdown fences."

        user_message = f"""Analyze this job application and return JSON with exactly these keys:
- tailored_resume: full rewritten resume text optimized for this role
- strengths: list of 3-6 bullet points where candidate strongly matches the JD
- gaps: list of 3-6 bullet points where candidate falls short of JD requirements
- similar_companies: list of 5 company names similar to {company} worth targeting
- similar_roles: list of 5 adjacent job titles worth applying for
- live_openings_queries: list of 3 search query strings to find similar live openings

Job Description:
{jd_text}

Resume:
{resume_text}

Target Company: {company}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
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
        raise ValueError(f"Error running prompt1: {e}")
