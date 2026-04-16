"""Pipeline orchestrator for job search copilot."""

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from memory import (
    init_db,
    upsert_job,
    get_job,
    upsert_interviewer,
    get_all_jobs_with_analyses,
)
from jd_fetcher import load_jd
from company_research import search_company, merge_with_override, build_company_profile_text
from interviewer import build_interviewer_context
from prompt1 import run_prompt1
from prompt2 import run_prompt2
from prompt3 import run_prompt3

# Default DB path
DB_PATH = str(Path(__file__).parent.parent / "data" / "memory" / "copilot.db")

def run_job_analysis(
    jd_source: str,
    resume_pdf_path: str,
    company: str,
    title: str,
    company_override: Optional[str] = None,
    db_path: str = DB_PATH,
    api_key: Optional[str] = None,
) -> str:
    """Analyze a job application.

    Args:
        jd_source: Job description URL or pasted text
        resume_pdf_path: Path to resume PDF
        company: Company name
        title: Job title
        company_override: Optional company research override
        db_path: Database path
        api_key: Anthropic API key

    Returns:
        job_id
    """
    # Initialize DB
    init_db(db_path)

    # 1. Load JD
    jd_text, portal = load_jd(jd_source)

    # 2. Extract resume text from PDF
    try:
        from pypdf import PdfReader

        reader = PdfReader(resume_pdf_path)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()
    except Exception as e:
        print(f"Warning: Failed to extract resume from {resume_pdf_path}: {e}")
        resume_text = ""

    # 3. Research company
    auto_profile = search_company(company, api_key)

    # 4. Merge with override
    company_profile = merge_with_override(auto_profile, company_override)

    # 5. Run prompt1 analysis
    analysis = run_prompt1(resume_text, jd_text, company, api_key)

    # 6. Generate job_id
    job_id = uuid.uuid4().hex

    # 7. Prepare job data
    jd_url = jd_source if jd_source.lower().startswith("http") else None

    job_data = {
        "job_id": job_id,
        "company": company,
        "title": title,
        "jd_text": jd_text,
        "jd_url": jd_url,
        "portal": portal,
        "gap_analysis": json.dumps(analysis),
        "tailored_resume": analysis.get("tailored_resume", ""),
        "company_competition": json.dumps(company_profile.get("competition", [])),
        "company_reputation": company_profile.get("reputation", ""),
        "company_alternatives": json.dumps(company_profile.get("alternatives", [])),
        "similar_companies": json.dumps(analysis.get("similar_companies", [])),
        "similar_roles": json.dumps(analysis.get("similar_roles", [])),
        "live_openings": json.dumps(analysis.get("live_openings_queries", [])),
    }

    # 8. Upsert job
    upsert_job(job_data, db_path)

    return job_id


def run_interview_prep(
    job_id: str,
    interviewer_name: str,
    role: str,
    seniority: str,
    linkedin_url: Optional[str] = None,
    additional_context: Optional[str] = None,
    system_prompt_override: Optional[str] = None,
    db_path: str = DB_PATH,
    api_key: Optional[str] = None,
) -> str:
    """Prepare interview materials for a specific job and interviewer.

    Args:
        job_id: Job ID from run_job_analysis
        interviewer_name: Interviewer name
        role: Interviewer role
        seniority: Interviewer seniority
        linkedin_url: Optional LinkedIn URL
        additional_context: Optional additional context
        system_prompt_override: Optional custom system prompt for prompt2
        db_path: Database path
        api_key: Anthropic API key

    Returns:
        interviewer_id

    Raises:
        ValueError: If job not found
    """
    # Initialize DB
    init_db(db_path)

    # 1. Get job
    job = get_job(job_id, db_path)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    # 2. Parse gap_analysis
    gap_analysis = {}
    if job.get("gap_analysis"):
        try:
            gap_analysis = json.loads(job["gap_analysis"])
        except (json.JSONDecodeError, TypeError):
            gap_analysis = {}

    # 3. Build company_profile
    company_profile = {
        "competition": [],
        "reputation": "",
        "alternatives": [],
    }

    if job.get("company_competition"):
        try:
            company_profile["competition"] = json.loads(job["company_competition"])
        except (json.JSONDecodeError, TypeError):
            pass

    if job.get("company_reputation"):
        company_profile["reputation"] = job["company_reputation"]

    if job.get("company_alternatives"):
        try:
            company_profile["alternatives"] = json.loads(job["company_alternatives"])
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Build interviewer context
    iv_context = build_interviewer_context(
        interviewer_name, role, seniority, linkedin_url, additional_context
    )

    # 5. Run prompt2
    qa_prep = run_prompt2(
        gap_analysis,
        job.get("tailored_resume", ""),
        company_profile,
        iv_context,
        system_prompt_override,
        api_key,
    )

    # 6. Generate interviewer_id
    interviewer_id = uuid.uuid4().hex

    # 7. Prepare interviewer data
    interviewer_data = {
        "interviewer_id": interviewer_id,
        "job_id": job_id,
        "name": interviewer_name,
        "role": role,
        "seniority": seniority,
        "linkedin_url": linkedin_url,
        "additional_context": additional_context,
        "system_prompt_override": system_prompt_override,
        "qa_prep": json.dumps(qa_prep),
    }

    # 8. Upsert interviewer
    upsert_interviewer(interviewer_data, db_path)

    return interviewer_id


def run_comparison(db_path: str = DB_PATH, api_key: Optional[str] = None) -> dict:
    """Compare all analyzed jobs and provide ranked recommendations.

    Args:
        db_path: Database path
        api_key: Anthropic API key

    Returns:
        Comparison dict with ranked_jobs, pattern_insights, recommended_next_steps
    """
    # Initialize DB
    init_db(db_path)

    # 1. Get all jobs with analyses
    jobs = get_all_jobs_with_analyses(db_path)

    if not jobs:
        return {
            "ranked_jobs": [],
            "pattern_insights": [],
            "recommended_next_steps": [],
        }

    # 2. Run prompt3
    comparison = run_prompt3(jobs, api_key)

    return comparison
