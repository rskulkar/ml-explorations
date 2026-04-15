"""Dashboard page."""

import sys
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, list_jobs, list_interviewers, upsert_job, get_all_jobs_with_analyses

st.set_page_config(page_title="Dashboard", layout="wide")

# Initialize database
db_path = Path(__file__).parent.parent / "data" / "memory" / "copilot.db"
init_db(str(db_path))

st.title("Dashboard")

# Get all jobs
jobs = list_jobs(status="active", db_path=str(db_path))
jobs_all = []

# Get all statuses
try:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        jobs_all.append(dict(row))
except Exception:
    jobs_all = jobs

# Show metrics
analyzed_jobs = get_all_jobs_with_analyses(str(db_path))
active_jobs = [j for j in jobs_all if j.get("status") == "active"]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Jobs", len(jobs_all))
with col2:
    st.metric("Active Jobs", len(active_jobs))
with col3:
    st.metric("Jobs with Analysis", len(analyzed_jobs))

# Show all jobs table
st.subheader("All Jobs")
if jobs_all:
    table_data = []
    for job in jobs_all:
        table_data.append({
            "Company": job.get("company", ""),
            "Title": job.get("title", ""),
            "Status": job.get("status", "active"),
            "Created": job.get("created_at", ""),
            "Job ID": job.get("job_id", "")[:8] + "...",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# Expandable job details
st.subheader("Job Details")
for job in jobs_all:
    with st.expander(f"{job['company']} — {job['title']}"):
        col1, col2 = st.columns(2)

        with col1:
            # Show gap analysis summary if available
            if job.get("gap_analysis"):
                st.subheader("Gap Analysis Summary")
                try:
                    gap_analysis = json.loads(job["gap_analysis"])

                    strengths = gap_analysis.get("strengths", [])
                    if strengths:
                        st.write("**Strengths:**")
                        for s in strengths[:3]:
                            st.write(f"• {s}")

                    gaps = gap_analysis.get("gaps", [])
                    if gaps:
                        st.write("**Gaps:**")
                        for g in gaps[:3]:
                            st.write(f"• {g}")
                except (json.JSONDecodeError, TypeError):
                    st.write("Could not parse gap analysis")

        with col2:
            # Show interviewers
            interviewers = list_interviewers(job["job_id"], str(db_path))
            st.subheader(f"Interviewers ({len(interviewers)})")
            if interviewers:
                for iv in interviewers:
                    st.write(f"• {iv['name']} ({iv['role']})")
            else:
                st.write("No interviewers added")

        # Status update
        current_status = job.get("status", "active")
        new_status = st.selectbox(
            "Update Status",
            ["active", "offer", "rejected", "withdrawn"],
            index=["active", "offer", "rejected", "withdrawn"].index(current_status),
            key=f"status_{job['job_id']}",
        )

        if new_status != current_status:
            # Update job status
            updated_job = job.copy()
            updated_job["status"] = new_status
            upsert_job(updated_job, str(db_path))
            st.success(f"Status updated to {new_status}")
            st.rerun()

st.markdown("---")
st.markdown(
    """
    **Tips:**
    - Use Add Job to fetch and analyze new opportunities
    - Use Add Interviewer to prepare for specific interviews
    - Use Compare Opportunities to rank all analyzed jobs
    """
)
