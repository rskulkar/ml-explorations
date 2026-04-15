"""SQLite-based memory system for job search copilot."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


def init_db(db_path: Optional[str] = None) -> str:
    """Initialize database with schema. Create parent dirs if needed.

    Args:
        db_path: Path to database file. Defaults to data/memory/copilot.db relative to project root.

    Returns:
        Path to the database file.
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            title TEXT NOT NULL,
            jd_text TEXT,
            jd_url TEXT,
            portal TEXT,
            resume_used TEXT,
            gap_analysis TEXT,
            tailored_resume TEXT,
            company_competition TEXT,
            company_reputation TEXT,
            company_alternatives TEXT,
            company_profile_override TEXT,
            similar_companies TEXT,
            similar_roles TEXT,
            live_openings TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'active'
        )
    """)

    # Create interviewers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interviewers (
            interviewer_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT,
            seniority TEXT,
            linkedin_url TEXT,
            linkedin_text TEXT,
            additional_context TEXT,
            system_prompt_override TEXT,
            qa_prep TEXT,
            interview_date TEXT,
            outcome TEXT,
            notes TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )
    """)

    conn.commit()
    conn.close()

    return db_path


def upsert_job(job: dict, db_path: Optional[str] = None) -> None:
    """Insert or replace a job record.

    Args:
        job: Dictionary with job_id (required) and other fields
        db_path: Path to database file. Defaults to data/memory/copilot.db relative to project root.
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    # Ensure all fields exist in job dict
    job_id = job.get("job_id")
    if not job_id:
        raise ValueError("job_id is required")

    # Convert structured fields to JSON strings
    for json_field in ["gap_analysis", "tailored_resume", "company_competition",
                       "company_reputation", "company_alternatives", "company_profile_override",
                       "similar_companies", "similar_roles", "live_openings"]:
        if json_field in job and isinstance(job[json_field], (dict, list)):
            job[json_field] = json.dumps(job[json_field])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all possible columns
    columns = [
        "job_id", "company", "title", "jd_text", "jd_url", "portal",
        "resume_used", "gap_analysis", "tailored_resume", "company_competition",
        "company_reputation", "company_alternatives", "company_profile_override",
        "similar_companies", "similar_roles", "live_openings", "status"
    ]

    # Build INSERT OR REPLACE statement
    values = [job.get(col) for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    cols_str = ", ".join(columns)

    cursor.execute(f"""
        INSERT OR REPLACE INTO jobs ({cols_str})
        VALUES ({placeholders})
    """, values)

    conn.commit()
    conn.close()


def get_job(job_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a job record by ID.

    Args:
        job_id: Job ID
        db_path: Path to database file

    Returns:
        Dictionary with job data or None if not found
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)

    # Parse JSON fields
    for json_field in ["gap_analysis", "tailored_resume", "company_competition",
                       "company_reputation", "company_alternatives", "company_profile_override",
                       "similar_companies", "similar_roles", "live_openings"]:
        if result.get(json_field):
            try:
                result[json_field] = json.loads(result[json_field])
            except (json.JSONDecodeError, TypeError):
                pass

    return result


def list_jobs(status: str = "active", db_path: Optional[str] = None) -> list[dict]:
    """List jobs by status.

    Args:
        status: Filter by status (default: 'active')
        db_path: Path to database file

    Returns:
        List of job dictionaries
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        result = dict(row)

        # Parse JSON fields
        for json_field in ["gap_analysis", "tailored_resume", "company_competition",
                           "company_reputation", "company_alternatives", "company_profile_override",
                           "similar_companies", "similar_roles", "live_openings"]:
            if result.get(json_field):
                try:
                    result[json_field] = json.loads(result[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass

        results.append(result)

    return results


def upsert_interviewer(iv: dict, db_path: Optional[str] = None) -> None:
    """Insert or replace an interviewer record.

    Args:
        iv: Dictionary with interviewer_id (required), job_id (required) and other fields
        db_path: Path to database file
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    interviewer_id = iv.get("interviewer_id")
    if not interviewer_id:
        raise ValueError("interviewer_id is required")

    if not iv.get("job_id"):
        raise ValueError("job_id is required")

    # Convert structured fields to JSON strings
    for json_field in ["qa_prep"]:
        if json_field in iv and isinstance(iv[json_field], (dict, list)):
            iv[json_field] = json.dumps(iv[json_field])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns = [
        "interviewer_id", "job_id", "name", "role", "seniority",
        "linkedin_url", "linkedin_text", "additional_context",
        "system_prompt_override", "qa_prep", "interview_date",
        "outcome", "notes"
    ]

    values = [iv.get(col) for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    cols_str = ", ".join(columns)

    cursor.execute(f"""
        INSERT OR REPLACE INTO interviewers ({cols_str})
        VALUES ({placeholders})
    """, values)

    conn.commit()
    conn.close()


def get_interviewer(interviewer_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get an interviewer record by ID.

    Args:
        interviewer_id: Interviewer ID
        db_path: Path to database file

    Returns:
        Dictionary with interviewer data or None if not found
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM interviewers WHERE interviewer_id = ?", (interviewer_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)

    # Parse JSON fields
    for json_field in ["qa_prep"]:
        if result.get(json_field):
            try:
                result[json_field] = json.loads(result[json_field])
            except (json.JSONDecodeError, TypeError):
                pass

    return result


def list_interviewers(job_id: str, db_path: Optional[str] = None) -> list[dict]:
    """List interviewers for a job.

    Args:
        job_id: Job ID
        db_path: Path to database file

    Returns:
        List of interviewer dictionaries
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM interviewers WHERE job_id = ?", (job_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        result = dict(row)

        # Parse JSON fields
        for json_field in ["qa_prep"]:
            if result.get(json_field):
                try:
                    result[json_field] = json.loads(result[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass

        results.append(result)

    return results


def get_all_jobs_with_analyses(db_path: Optional[str] = None) -> list[dict]:
    """Get all jobs that have gap analysis.

    Args:
        db_path: Path to database file

    Returns:
        List of job dictionaries with gap_analysis
    """
    if db_path is None:
        db_path = str(Path(__file__).parent.parent.parent / "data" / "memory" / "copilot.db")

    db_path = str(Path(db_path))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE gap_analysis IS NOT NULL ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        result = dict(row)

        # Parse JSON fields
        for json_field in ["gap_analysis", "tailored_resume", "company_competition",
                           "company_reputation", "company_alternatives", "company_profile_override",
                           "similar_companies", "similar_roles", "live_openings"]:
            if result.get(json_field):
                try:
                    result[json_field] = json.loads(result[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass

        results.append(result)

    return results
