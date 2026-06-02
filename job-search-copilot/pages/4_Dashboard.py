"""Dashboard page."""

import sys
import os
import json
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, list_interviewers, upsert_job

st.set_page_config(page_title="Dashboard", layout="wide")

db_path = str(Path(__file__).parent.parent / "data" / "memory" / "copilot.db")
init_db(db_path)

STATUSES = ["active", "offer", "rejected", "withdrawn"]


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
    """fit = strengths / (strengths + gaps). Returns 0.0 if no data."""
    strengths = len(gap_analysis.get("strengths") or [])
    gaps = len(gap_analysis.get("gaps") or [])
    total = strengths + gaps
    if total == 0:
        return 0.0
    return round(strengths / total * 100, 1)


# Load all jobs directly
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT * FROM jobs ORDER BY created_at DESC")
jobs_all = [dict(row) for row in cur.fetchall()]
conn.close()

st.title("Dashboard")

# ── Metrics ──────────────────────────────────────────────────────────────────
active_jobs = [j for j in jobs_all if (j.get("status") or "active") == "active"]
analysed_jobs = [j for j in jobs_all if j.get("gap_analysis")]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Jobs", len(jobs_all))
with col2:
    st.metric("Active Jobs", len(active_jobs))
with col3:
    st.metric("Jobs with Analysis", len(analysed_jobs))

st.divider()

# ── Ranked jobs table ────────────────────────────────────────────────────────
st.subheader("All Jobs — ranked by fit score")

if jobs_all:
    table_rows = []
    for job in jobs_all:
        analysis = safe_json(job.get("gap_analysis"))
        fit = compute_fit_score(analysis)
        table_rows.append({
            "_fit": fit,
            "_created": job.get("created_at") or "",
            "Company": job.get("company") or "",
            "Title": job.get("title") or "",
            "Status": (job.get("status") or "active").capitalize(),
            "Fit Score": f"{fit}%" if fit > 0 else "—",
            "Strengths": len(analysis.get("strengths") or []),
            "Gaps": len(analysis.get("gaps") or []),
            "Created": (job.get("created_at") or "")[:10],
            "_job_id": job.get("job_id"),
        })

    # Sort: fit score descending, created_at ascending for ties
    table_rows.sort(key=lambda r: (-r["_fit"], r["_created"]))

    # Add rank
    for i, row in enumerate(table_rows):
        row["Rank"] = i + 1

    df = pd.DataFrame(table_rows)[["Rank", "Company", "Title", "Status", "Fit Score", "Strengths", "Gaps", "Created"]]
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No jobs saved yet.")

st.divider()

# ── Job details ───────────────────────────────────────────────────────────────
st.subheader("Job details")

# Re-sort jobs_all by fit score for the expanders too
jobs_ranked = sorted(
    jobs_all,
    key=lambda j: (
        -compute_fit_score(safe_json(j.get("gap_analysis"))),
        j.get("created_at") or ""
    )
)

for i, job in enumerate(jobs_ranked):
    analysis = safe_json(job.get("gap_analysis"))
    fit = compute_fit_score(analysis)
    status = (job.get("status") or "active")
    if status not in STATUSES:
        status = "active"
    label = f"#{i+1}  {job.get('company','')} — {job.get('title','')}  |  {status.upper()}  |  Fit: {fit}%"

    with st.expander(label, expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if analysis:
                st.markdown("**Strengths**")
                for s in (analysis.get("strengths") or [])[:3]:
                    st.markdown(f"- {s}")
                st.markdown("**Gaps**")
                for g in (analysis.get("gaps") or [])[:3]:
                    st.markdown(f"- {g}")
            else:
                st.info("No analysis yet.")

        with col2:
            interviewers = list_interviewers(job["job_id"], db_path)
            st.markdown(f"**Interviewers ({len(interviewers)})**")
            if interviewers:
                for iv in interviewers:
                    st.write(f"- {iv['name']} ({iv.get('role','')})")
            else:
                st.write("None added yet.")

        new_status = st.selectbox(
            "Status",
            STATUSES,
            index=STATUSES.index(status),
            key=f"status_{job['job_id']}"
        )
        if new_status != status:
            updated = dict(job)
            updated["status"] = new_status
            upsert_job(updated, db_path)
            st.success(f"Status updated to {new_status}")
            st.rerun()