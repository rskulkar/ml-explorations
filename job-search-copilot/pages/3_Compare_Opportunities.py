"""Compare Opportunities page."""

import sys
import os
import json
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, get_all_jobs_with_analyses
from pipeline import run_comparison

st.set_page_config(page_title="Compare Opportunities", layout="wide")

api_key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not configured.")
    st.stop()

db_path = str(Path(__file__).parent.parent / "data" / "memory" / "copilot.db")
init_db(db_path)

COMPARISON_CACHE = Path(__file__).parent.parent / "data" / "memory" / "last_comparison.json"


def safe_json(value):
    if not value:
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def compute_fit_score(gap_analysis: dict) -> float:
    strengths = len(gap_analysis.get("strengths") or [])
    gaps = len(gap_analysis.get("gaps") or [])
    total = strengths + gaps
    return round(strengths / total * 100, 1) if total > 0 else 0.0


def save_comparison(comparison: dict):
    COMPARISON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_CACHE, "w") as f:
        json.dump(comparison, f, indent=2)


def load_comparison() -> dict | None:
    if COMPARISON_CACHE.exists():
        try:
            with open(COMPARISON_CACHE) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def build_deterministic_ranking(jobs: list) -> list:
    """Rank jobs by fit score (deterministic). Returns list of dicts."""
    rows = []
    for job in jobs:
        analysis = safe_json(job.get("gap_analysis"))
        fit = compute_fit_score(analysis)
        rows.append({
            "job_id": job.get("job_id"),
            "company": job.get("company", ""),
            "title": job.get("title", ""),
            "fit_score": fit,
            "created_at": job.get("created_at") or "",
        })
    rows.sort(key=lambda r: (-r["fit_score"], r["created_at"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def display_comparison(comparison: dict, ranked_rows: list):
    # Merge LLM rationale into deterministic ranking
    rationale_map = {
        r.get("job_id"): r.get("rationale", "")
        for r in comparison.get("ranked_jobs", [])
    }

    table = []
    for r in ranked_rows:
        table.append({
            "Rank": r["rank"],
            "Company": r["company"],
            "Title": r["title"],
            "Fit Score": f"{r['fit_score']}%",
            "Rationale": rationale_map.get(r["job_id"], "—"),
        })

    st.subheader("Ranked Opportunities")
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("Strategic Insights")
    for i, insight in enumerate(comparison.get("pattern_insights", []), 1):
        st.write(f"{i}. {insight}")

    st.subheader("Recommended Next Steps")
    for i, step in enumerate(comparison.get("recommended_next_steps", []), 1):
        st.write(f"{i}. {step}")


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Compare Opportunities")

jobs = get_all_jobs_with_analyses(db_path)
st.metric("Jobs with analysis", len(jobs))

if len(jobs) < 2:
    st.warning("Add at least 2 analysed jobs to compare.")
    st.stop()

ranked_rows = build_deterministic_ranking(jobs)

# Load cached comparison if available
cached = load_comparison()

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("Run Comparison")
with col2:
    if cached:
        st.caption("Showing last saved comparison. Click 'Run Comparison' to refresh.")

if run_btn:
    try:
        with st.spinner("Comparing opportunities..."):
            comparison = run_comparison(db_path, api_key)
        save_comparison(comparison)
        st.session_state["last_comparison"] = comparison
        display_comparison(comparison, ranked_rows)
    except Exception as e:
        st.error(f"Error comparing opportunities: {e}")

elif "last_comparison" in st.session_state:
    display_comparison(st.session_state["last_comparison"], ranked_rows)

elif cached:
    display_comparison(cached, ranked_rows)