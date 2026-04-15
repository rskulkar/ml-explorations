"""Company research using Claude Haiku."""

import json
import os
import re
from typing import Optional


def search_company(company: str, api_key: Optional[str] = None) -> dict:
    """Research a company using Claude Haiku.

    Args:
        company: Company name to research
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.

    Returns:
        Dictionary with competition, reputation, alternatives keys.
        On failure returns {"competition": [], "reputation": "Not available", "alternatives": []}
    """
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return {"competition": [], "reputation": "Not available", "alternatives": []}

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        system_prompt = "You are a business intelligence analyst. Research the given company and return structured JSON only, no markdown fences."

        user_message = f"""Research this company and return JSON with exactly these keys:
- competition: list of 5 competitor company names
- reputation: 2-3 sentence summary of company reputation, culture, and market position
- alternatives: list of 5 similar companies where a candidate might also apply

Company: {company}"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
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

    except Exception as e:
        print(f"Error researching company {company}: {e}")
        return {"competition": [], "reputation": "Not available", "alternatives": []}


def merge_with_override(auto_profile: dict, override_text: Optional[str]) -> dict:
    """Merge auto-generated profile with user override.

    Args:
        auto_profile: Auto-generated company profile dict
        override_text: User-provided override text, or None

    Returns:
        Merged profile dict with optional user_notes key
    """
    if not override_text or not override_text.strip():
        return auto_profile

    result = auto_profile.copy()
    result["user_notes"] = override_text.strip()
    return result


def build_company_profile_text(profile: dict) -> str:
    """Build formatted company profile text.

    Args:
        profile: Company profile dict with competition, reputation, alternatives, optional user_notes

    Returns:
        Formatted text representation
    """
    competition = profile.get("competition", [])
    reputation = profile.get("reputation", "")
    alternatives = profile.get("alternatives", [])
    user_notes = profile.get("user_notes")

    competition_str = ", ".join(competition) if competition else "None available"
    alternatives_str = ", ".join(alternatives) if alternatives else "None available"

    text = f"Competition: {competition_str}\nReputation: {reputation}\nAlternatives: {alternatives_str}"

    if user_notes:
        text += f"\nNotes: {user_notes}"

    return text
