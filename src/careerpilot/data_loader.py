"""Dataset loaders for curated job postings and sample resumes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from careerpilot.config import JOBS_CSV_PATH, SAMPLE_RESUMES_DIR
from careerpilot.models import JobPosting


def _split_skills(raw: object) -> list[str]:
    """Split a semicolon-delimited skills string into clean skill names."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    return [item.strip() for item in str(raw).split(";") if item.strip()]


def load_jobs(csv_path: Path | None = None) -> list[JobPosting]:
    """Load and validate job postings from CSV using Pandas and Pydantic."""
    path = csv_path or JOBS_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Job dataset not found at {path}")

    df = pd.read_csv(path)
    jobs: list[JobPosting] = []
    for record in df.to_dict(orient="records"):
        jobs.append(
            JobPosting(
                job_id=str(record["job_id"]).strip(),
                title=str(record["title"]).strip(),
                company=str(record["company"]).strip(),
                location=str(record["location"]).strip(),
                work_mode=str(record["work_mode"]).strip(),
                experience_level=str(record["experience_level"]).strip(),
                salary_min=int(record["salary_min"]),
                salary_max=int(record["salary_max"]),
                industry=str(record["industry"]).strip(),
                required_skills=_split_skills(record.get("required_skills")),
                nice_to_have_skills=_split_skills(record.get("nice_to_have_skills")),
                description=str(record["description"]).strip(),
                posted_days_ago=int(record.get("posted_days_ago", 1)),
            )
        )
    return jobs


def jobs_to_dataframe(jobs: list[JobPosting]) -> pd.DataFrame:
    """Convert validated JobPosting models into a Pandas DataFrame for analytics."""
    rows = [
        {
            "job_id": j.job_id,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "work_mode": j.work_mode,
            "experience_level": j.experience_level,
            "salary_min": j.salary_min,
            "salary_max": j.salary_max,
            "industry": j.industry,
            "required_skills": ", ".join(j.required_skills),
            "nice_to_have_skills": ", ".join(j.nice_to_have_skills),
            "posted_days_ago": j.posted_days_ago,
        }
        for j in jobs
    ]
    return pd.DataFrame(rows)


def load_sample_resumes(resumes_dir: Path | None = None) -> dict[str, str]:
    """Load sample resume text files keyed by friendly display label."""
    directory = resumes_dir or SAMPLE_RESUMES_DIR
    if not directory.exists():
        return {}

    samples: dict[str, str] = {}
    for file_path in sorted(directory.glob("*.txt")):
        label = file_path.stem.replace("_", " ").title()
        samples[label] = file_path.read_text(encoding="utf-8")
    return samples
