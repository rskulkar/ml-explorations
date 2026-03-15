"""Claude Haiku compliance analysis agent for the cybersecurity-agent pipeline."""
from __future__ import annotations

import json

import anthropic

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """\
You are a cybersecurity compliance expert specialising in ISO 27001/27002 and NIST \
Cybersecurity Framework gap analysis. You will be given an organisational security \
control and relevant excerpts from a security standard. Your task is to assess the \
control against the standard and return a structured JSON gap analysis.

Return ONLY valid JSON with exactly these keys:
- "compliance_level": one of "compliant", "partial", or "non-compliant"
- "gaps": a list of strings describing specific gaps or missing elements
- "recommendations": a list of strings with concrete remediation steps

Do not include any text outside the JSON object.\
"""


def analyse_control(
    control_text: str,
    standard_chunks: list[str],
) -> dict:
    """Analyse an organisational security control against standard excerpts.

    Args:
        control_text: The full text of the organisational control to assess.
        standard_chunks: Relevant excerpts from the security standard (from RAG retrieval).

    Returns:
        Dict with keys: compliance_level (str), gaps (list[str]), recommendations (list[str]).

    Raises:
        ValueError: If the Claude response cannot be parsed as JSON.
    """
    standard_context = "\n\n---\n\n".join(
        f"[Standard excerpt {i + 1}]\n{chunk}"
        for i, chunk in enumerate(standard_chunks)
    )

    user_message = (
        f"ORGANISATIONAL CONTROL:\n{control_text}\n\n"
        f"RELEVANT STANDARD EXCERPTS:\n{standard_context}\n\n"
        "Perform a gap analysis and return the result as JSON."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse Claude response as JSON. Raw response: {raw!r}"
        ) from exc
