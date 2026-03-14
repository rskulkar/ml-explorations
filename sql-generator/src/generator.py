"""NL-to-SQL via Claude API. Requires ANTHROPIC_API_KEY env var."""
from __future__ import annotations
import re
import anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1000

SYSTEM_PROMPT = """\
You are an expert SQL query generator. Given a database schema and a natural language question,
write a single correct SQL query that answers the question.
Output ONLY the SQL query — no explanations, no markdown, no comments.
Use SQLite-compatible syntax.
If the question cannot be answered from the schema, return:
SELECT 'Cannot answer this question' AS message;
"""


def generate_sql(question: str, schema_context: str) -> str:
    """Call Claude API and return a SQL query string (markdown fences stripped)."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Database schema:\n{schema_context}\n\nQuestion: {question}\n\nSQL query:",
            }
        ],
    )
    return _strip_markdown(response.content[0].text)


def _strip_markdown(text: str) -> str:
    m = re.match(r"^```(?:sql)?\s*\n(.*?)\n?```\s*$", text.strip(), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()
