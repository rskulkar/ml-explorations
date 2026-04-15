"""Compare Opportunities page."""

import sys
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, get_all_jobs_with_analyses
from pipeline import run_comparison

st.set_page_config(page_title="Compare Opportunities", layout="wide")

# Get API key
api_key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not configured. Please set ANTHROPIC_API_KEY or enter it in the sidebar.")
    st.stop()

# Initialize database
db_path = Path(__file__).parent.parent / "data" / "memory" / "copilot.db"
init_db(str(db_path))

st.title("Compare Opportunities")

# Get analyzed jobs
jobs = get_all_jobs_with_analyses(str(db_path))

# Show metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Jobs with analysis", len(jobs))

# Show summary table
if jobs:
    summary_data = []
    for job in jobs:
        summary_data.append({
            "Company": job.get("company", ""),
            "Title": job.get("title", ""),
            "Status": job.get("status", "active"),
            "Created": job.get("created_at", ""),
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# Check if we have enough jobs
if len(jobs) < 2:
    st.warning("Add at least 2 analysed jobs to compare.")
    st.stop()

# Run comparison
if st.button("Run Comparison"):
    try:
        with st.spinner("Comparing opportunities..."):
            comparison = run_comparison(str(db_path), api_key)

        # Display ranked jobs
        st.subheader("Ranked Opportunities")
        ranked_jobs = comparison.get("ranked_jobs", [])
        if ranked_jobs:
            ranked_df = pd.DataFrame(ranked_jobs)
            st.dataframe(ranked_df, use_container_width=True)
        else:
            st.write("No ranked jobs generated")

        # Display strategic insights
        st.subheader("Strategic Insights")
        pattern_insights = comparison.get("pattern_insights", [])
        if pattern_insights:
            for i, insight in enumerate(pattern_insights, 1):
                st.write(f"{i}. {insight}")
        else:
            st.write("No insights generated")

        # Display recommended next steps
        st.subheader("Recommended Next Steps")
        recommended_next_steps = comparison.get("recommended_next_steps", [])
        if recommended_next_steps:
            for i, step in enumerate(recommended_next_steps, 1):
                st.write(f"{i}. {step}")
        else:
            st.write("No next steps recommended")

    except Exception as e:
        st.error(f"Error comparing opportunities: {e}")
