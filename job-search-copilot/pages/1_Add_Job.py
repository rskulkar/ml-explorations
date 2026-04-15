"""Add Job Opportunity page."""

import sys
import os
import json
import tempfile
from pathlib import Path

import streamlit as st

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, list_jobs
from pipeline import run_job_analysis

st.set_page_config(page_title="Add Job", layout="wide")

# Get API key
api_key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not configured. Please set ANTHROPIC_API_KEY or enter it in the sidebar.")
    st.stop()

# Initialize database
db_path = Path(__file__).parent.parent / "data" / "memory" / "copilot.db"
init_db(str(db_path))

st.title("Add Job Opportunity")

# Form fields
st.subheader("Job Details")

jd_source = st.text_area("Job Description URL or paste JD text", height=150)
company = st.text_input("Company Name")
title = st.text_input("Job Title")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
company_override = st.text_area("Additional company notes (optional)", height=80)

# Validate and process
if st.button("Analyse Job"):
    # Validation
    if not jd_source.strip():
        st.error("Please provide a job description URL or paste JD text")
        st.stop()

    if not company.strip():
        st.error("Please enter company name")
        st.stop()

    if not title.strip():
        st.error("Please enter job title")
        st.stop()

    if not resume_file:
        st.error("Please upload a resume PDF")
        st.stop()

    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resume_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Analysing job opportunity..."):
            job_id = run_job_analysis(
                jd_source=jd_source,
                resume_pdf_path=tmp_path,
                company=company,
                title=title,
                company_override=company_override if company_override.strip() else None,
                db_path=str(db_path),
                api_key=api_key,
            )

        # Store job_id in session state
        st.session_state["last_job_id"] = job_id

        # Retrieve the saved job to display results
        from memory import get_job

        job = get_job(job_id, str(db_path))

        if job:
            # Parse analyses
            gap_analysis = {}
            if job.get("gap_analysis"):
                try:
                    gap_analysis = json.loads(job["gap_analysis"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Display results in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Strengths")
                strengths = gap_analysis.get("strengths", [])
                if strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
                else:
                    st.write("No strengths identified")

                st.subheader("Gaps")
                gaps = gap_analysis.get("gaps", [])
                if gaps:
                    for gap in gaps:
                        st.write(f"• {gap}")
                else:
                    st.write("No gaps identified")

            with col2:
                st.subheader("Similar Companies")
                similar_companies = gap_analysis.get("similar_companies", [])
                if similar_companies:
                    for comp in similar_companies:
                        st.write(f"• {comp}")
                else:
                    st.write("No similar companies identified")

                st.subheader("Similar Roles")
                similar_roles = gap_analysis.get("similar_roles", [])
                if similar_roles:
                    for role in similar_roles:
                        st.write(f"• {role}")
                else:
                    st.write("No similar roles identified")

            # Tailored resume
            with st.expander("Tailored Resume"):
                tailored = job.get("tailored_resume", "")
                if tailored:
                    st.text(tailored)
                else:
                    st.write("No tailored resume generated")

            st.success(f"✓ Job saved — ID: {job_id}")

    except Exception as e:
        st.error(f"Error analysing job: {e}")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
