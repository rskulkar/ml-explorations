"""Add Interviewer page."""

import sys
import os
from pathlib import Path

import streamlit as st

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import init_db, list_jobs
from pipeline import run_interview_prep

st.set_page_config(page_title="Add Interviewer", layout="wide")

# Get API key
api_key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("API key not configured. Please set ANTHROPIC_API_KEY or enter it in the sidebar.")
    st.stop()

# Initialize database
db_path = Path(__file__).parent.parent / "data" / "memory" / "copilot.db"
init_db(str(db_path))

st.title("Add Interviewer")

# Load jobs
jobs = list_jobs(db_path=str(db_path))

if not jobs:
    st.warning("No jobs added yet. Go to Add Job first.")
    st.stop()

# Job selection
job_options = {f"{j['company']} — {j['title']}": j['job_id'] for j in jobs}
selected_label = st.selectbox("Select Job", list(job_options.keys()))
job_id = job_options[selected_label]

st.subheader("Interviewer Details")

# Form fields
name = st.text_input("Interviewer Name")
role = st.text_input("Interviewer Role (e.g. Hiring Manager, CTO)")
seniority = st.selectbox(
    "Seniority",
    ["Junior", "Mid", "Senior", "Director", "VP", "C-Suite"],
)
linkedin_url = st.text_input("LinkedIn URL (optional)")
additional_context = st.text_area("Additional context about interviewer (optional)", height=100)
system_prompt_override = st.text_area(
    "Custom system prompt (optional — overrides default tone)",
    height=80,
)

# Process
if st.button("Generate Q&A Prep"):
    # Validation
    if not name.strip():
        st.error("Please enter interviewer name")
        st.stop()

    if not role.strip():
        st.error("Please enter interviewer role")
        st.stop()

    try:
        with st.spinner("Generating interview prep..."):
            interviewer_id = run_interview_prep(
                job_id=job_id,
                interviewer_name=name,
                role=role,
                seniority=seniority,
                linkedin_url=linkedin_url if linkedin_url.strip() else None,
                additional_context=additional_context if additional_context.strip() else None,
                system_prompt_override=system_prompt_override if system_prompt_override.strip() else None,
                db_path=str(db_path),
                api_key=api_key,
            )

        # Retrieve the saved interviewer prep
        from memory import get_interviewer
        import json

        interviewer = get_interviewer(interviewer_id, str(db_path))

        if interviewer:
            # Parse QA prep
            qa_prep = {}
            if interviewer.get("qa_prep"):
                try:
                    qa_prep = json.loads(interviewer["qa_prep"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Display STAR questions
            st.subheader("Behavioural Questions (STAR)")
            behavioural_star = qa_prep.get("behavioural_star", [])
            if behavioural_star:
                for item in behavioural_star:
                    with st.expander(item.get("question", "Question")):
                        st.write(f"**Situation:** {item.get('situation', '')}")
                        st.write(f"**Task:** {item.get('task', '')}")
                        st.write(f"**Action:** {item.get('action', '')}")
                        st.write(f"**Result:** {item.get('result', '')}")
            else:
                st.write("No behavioural questions generated")

            # Display technical questions
            st.subheader("Technical Questions")
            technical_questions = qa_prep.get("technical_questions", [])
            if technical_questions:
                for item in technical_questions:
                    with st.expander(item.get("question", "Question")):
                        st.write(f"**Answer:** {item.get('ideal_answer', '')}")
            else:
                st.write("No technical questions generated")

            # Display follow-up probes
            st.subheader("Follow-up Probes")
            follow_up_probes = qa_prep.get("follow_up_probes", [])
            if follow_up_probes:
                for probe in follow_up_probes:
                    st.write(f"• {probe}")
            else:
                st.write("No follow-up probes generated")

            # Tone notes
            tone_notes = qa_prep.get("tone_notes", "")
            st.info(f"**Tone notes:** {tone_notes}")

            st.success("✓ Interview prep saved")

    except Exception as e:
        st.error(f"Error generating interview prep: {e}")
