"""Interviewer context and LinkedIn profile handling."""

import requests
from typing import Optional


def fetch_linkedin_text(linkedin_url: str, timeout: int = 15) -> str:
    """Fetch LinkedIn profile text.

    Note: LinkedIn actively blocks scrapers. Use additional_context as the primary input path.

    Args:
        linkedin_url: LinkedIn profile URL
        timeout: Request timeout in seconds

    Returns:
        Profile text, or empty string on any exception or 403 response
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(linkedin_url, timeout=timeout, headers=headers)

        if response.status_code == 403:
            return ""

        response.raise_for_status()
        return response.text[:2000]  # Return first 2000 chars to avoid token limits

    except Exception:
        return ""


def build_interviewer_context(
    name: str,
    role: str,
    seniority: str,
    linkedin_url: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> str:
    """Build formatted interviewer context string.

    Args:
        name: Interviewer name
        role: Interviewer role/title
        seniority: Seniority level (e.g., "senior", "staff")
        linkedin_url: Optional LinkedIn profile URL
        additional_context: Optional manually provided context

    Returns:
        Formatted interviewer context string
    """
    lines = [f"Interviewer: {name}", f"Role: {role}", f"Seniority: {seniority}"]

    # Try LinkedIn first, then fall back to additional_context
    background = ""
    if linkedin_url:
        background = fetch_linkedin_text(linkedin_url)

    if not background and additional_context:
        background = additional_context

    if not background:
        background = "No additional context provided"

    lines.append(f"Background: {background}")

    return "\n".join(lines)
