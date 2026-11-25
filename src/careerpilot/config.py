"""Application configuration and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
JOBS_CSV_PATH = DATA_DIR / "jobs.csv"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
DEFAULT_STATE_PATH = PROJECT_ROOT / ".careerpilot_state.json"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for CareerPilot AI."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"
    jobs_csv_path: Path = JOBS_CSV_PATH

    @classmethod
    def from_env(cls, api_key_override: str | None = None) -> Settings:
        """Create settings from environment variables with optional UI override."""
        key = (api_key_override or os.getenv("OPENAI_API_KEY") or "").strip() or None
        return cls(
            openai_api_key=key,
            openai_model=os.getenv("CAREERPILOT_OPENAI_MODEL", "gpt-4o-mini").strip()
            or "gpt-4o-mini",
            embedding_model=os.getenv("CAREERPILOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
            or "all-MiniLM-L6-v2",
            jobs_csv_path=JOBS_CSV_PATH,
        )
