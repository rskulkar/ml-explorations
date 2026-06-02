"""Add Job Opportunity page."""

import sys
import os
import json
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, get_job, list_jobs, upsert_job
from pipeline import run_job_analysis
from prompt1 import run_prompt1
from pypdf import PdfReader

st.set_page_config(page_title="Add Job", layout="wide")

api_key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not configured. Add it in the sidebar on the home page.")
    st.stop()

db_path = str(Path(__file__).parent.parent / "data" / "memory" / "copilot.db")
init_db(db_path)

STATUSES = ["active", "offer", "rejected", "withdrawn"]


def safe_json(value):
    """Safely parse a value that may be a JSON string, dict, list, or None."""
    if not value:
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def display_analysis(analysis: dict):
    if not analysis:
        st.info("No analysis available.")
        return

    # Row 1: Strengths | Gaps
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strengths")
        for s in analysis.get("strengths") or []:
            st.markdown(f"- {s}")
    with col2:
        st.subheader("Gaps")
        for g in analysis.get("gaps") or []:
            st.markdown(f"- {g}")

    # Row 2: Similar Companies | Similar Roles
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Similar Companies")
        companies = safe_json(analysis.get("similar_companies"))
        if isinstance(companies, dict):
            companies = list(companies.values())
        for c in (companies if isinstance(companies, list) else []):
            st.markdown(f"- {c}")
    with col4:
        st.subheader("Similar Roles")
        roles = safe_json(analysis.get("similar_roles"))
        if isinstance(roles, dict):
            roles = list(roles.values())
        for r in (roles if isinstance(roles, list) else []):
            st.markdown(f"- {r}")

    with st.expander("Tailored Resume"):
        st.text(analysis.get("tailored_resume") or "")


st.title("Job Opportunities")

# ── SECTION 1: Saved jobs ────────────────────────────────────────────────────
jobs = list_jobs(status=None, db_path=db_path)

if jobs:
    st.header(f"Saved jobs ({len(jobs)})")
    for job in jobs:
        status = job.get("status") or "active"
        if status not in STATUSES:
            status = "active"
        label = f"{job['company']} — {job['title']}  |  {status.upper()}  |  {job['created_at'][:10]}"
        with st.expander(label, expanded=False):
            gap = job.get("gap_analysis")
            if gap:
                display_analysis(safe_json(gap))
            else:
                st.info("No analysis yet — re-analyse below to generate one.")

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
                st.success("Status updated.")
                st.rerun()
else:
    st.info("No jobs saved yet. Add your first job below.")

st.divider()

# ── SECTION 2: Add new job ───────────────────────────────────────────────────
st.header("Add new job")

with st.form("add_job_form"):
    jd_source = st.text_area("Job Description URL or paste JD text", height=150)
    col1, col2 = st.columns(2)
    with col1:
        company = st.text_input("Company Name")
    with col2:
        title = st.text_input("Job Title")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    company_override = st.text_area("Additional company notes (optional)", height=60)
    submitted = st.form_submit_button("Analyse & Save Job")

if submitted:
    if not jd_source or not company or not title or not resume_file:
        st.warning("Please fill in all required fields and upload your resume.")
    else:
        existing = [j for j in (jobs or [])
                    if (j.get("company") or "").lower().strip() == company.lower().strip()
                    and (j.get("title") or "").lower().strip() == title.lower().strip()]
        if existing:
            st.warning(f"Job already exists: {existing[0]['company']} — {existing[0]['title']} (ID: {existing[0]['job_id'][:8]}). Use Re-analyse below to update.")
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resume_file.read())
                    tmp_path = tmp.name

                with st.spinner("Analysing..."):
                    job_id = run_job_analysis(
                        jd_source, tmp_path, company, title,
                        company_override or None, db_path, api_key
                    )

                st.session_state["last_job_id"] = job_id
                job = get_job(job_id, db_path)
                display_analysis(safe_json(job.get("gap_analysis") if job else None))
                st.success(f"Job saved — ID: {job_id}")
                st.rerun()

            except Exception as e:
                st.error(f"Error analysing job: {e}")
                raise e

st.divider()

# ── SECTION 3: Re-analyse existing job ──────────────────────────────────────
st.header("Re-analyse existing job")

if not jobs:
    st.info("No saved jobs to re-analyse.")
else:
    job_options = {
        f"{j.get('company','?')} — {j.get('title','?')} ({j['job_id'][:8]})": j["job_id"]
        for j in jobs
    }
    selected_label = st.selectbox("Select job", list(job_options.keys()))
    job_id = job_options[selected_label]
    job = get_job(job_id, db_path)

    resume_file2 = st.file_uploader("Upload updated Resume (PDF)", type=["pdf"], key="reanalyse_resume")
    company_override2 = st.text_area(
        "Update company notes (optional)",
        value=job.get("company_profile_override") or "",
        height=60,
        key="reanalyse_override"
    )

    if st.button("Re-analyse"):
        if not resume_file2:
            st.warning("Please upload your resume PDF.")
        else:
            jd_text = job.get("jd_text") or ""
            if not jd_text:
                st.error("No JD text stored for this job. Please delete and re-add it.")
                st.stop()
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resume_file2.read())
                    tmp_path = tmp.name

                reader = PdfReader(tmp_path)
                resume_text = "\n".join(page.extract_text() or "" for page in reader.pages)

                with st.spinner("Re-analysing..."):
                    analysis = run_prompt1(resume_text, jd_text, job.get("company", ""), api_key)

                updated = dict(job)
                updated["gap_analysis"] = json.dumps(analysis)
                updated["tailored_resume"] = analysis.get("tailored_resume", "")
                updated["similar_companies"] = json.dumps(analysis.get("similar_companies", []))
                updated["similar_roles"] = json.dumps(analysis.get("similar_roles", []))
                updated["live_openings"] = json.dumps(analysis.get("live_openings_queries", []))
                if company_override2:
                    updated["company_profile_override"] = company_override2
                upsert_job(updated, db_path)

                display_analysis(safe_json(analysis))
                st.success(f"Analysis updated — ID: {job_id}")
                st.rerun()

            except Exception as e:
                st.error(f"Error re-analysing: {e}")
                raise e