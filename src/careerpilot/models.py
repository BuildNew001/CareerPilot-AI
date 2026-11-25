"""Pydantic domain models for resumes, preferences, jobs, matches, and agent turns."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    """Structured candidate profile extracted from a resume."""

    candidate_name: str = Field(default="Candidate", description="Candidate full name if present")
    current_title: str = Field(default="Software / AI Professional", description="Current or target job title")
    years_of_experience: float = Field(default=0.0, ge=0.0, description="Total estimated years of experience")
    seniority_level: Literal["Entry", "Mid", "Senior", "Lead", "Executive"] = Field(
        default="Mid", description="Inferred seniority tier"
    )
    skills: list[str] = Field(default_factory=list, description="Technical and core professional skills")
    tools_and_frameworks: list[str] = Field(
        default_factory=list, description="Libraries, platforms, cloud, and frameworks"
    )
    domains: list[str] = Field(default_factory=list, description="Domain areas such as NLP, FinTech, MLOps")
    education: list[str] = Field(default_factory=list, description="Degrees or certifications")
    career_interests: list[str] = Field(
        default_factory=list, description="Target roles or career interests"
    )
    summary: str = Field(default="", description="Concise 2-3 sentence professional summary")
    raw_text: str = Field(default="", description="Original extracted resume text")
    extraction_mode: Literal["openai", "fallback"] = Field(
        default="fallback", description="Whether OpenAI structured output or local extractor was used"
    )

    def all_skills(self) -> list[str]:
        """Return deduplicated combined skills and tools preserving order."""
        seen: set[str] = set()
        combined: list[str] = []
        for item in [*self.skills, *self.tools_and_frameworks]:
            normalized = item.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                combined.append(normalized)
        return combined

    def to_search_query(self) -> str:
        """Build a dense semantic search text representation of the candidate profile."""
        parts = [
            self.current_title,
            self.summary,
            "Skills: " + ", ".join(self.all_skills()),
            "Domains: " + ", ".join(self.domains),
            "Interests: " + ", ".join(self.career_interests),
        ]
        return " | ".join(p for p in parts if p and not p.endswith(": "))


class UserPreferences(BaseModel):
    """Persisted candidate job-search preferences and conversation memory."""

    preferred_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    work_modes: list[Literal["Remote", "Hybrid", "Onsite"]] = Field(default_factory=list)
    min_salary: int | None = Field(default=None, ge=0)
    experience_levels: list[str] = Field(default_factory=list)
    preferred_industries: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class JobPosting(BaseModel):
    """Validated job posting record from the job dataset."""

    job_id: str
    title: str
    company: str
    location: str
    work_mode: Literal["Remote", "Hybrid", "Onsite"]
    experience_level: Literal["Entry", "Mid", "Senior", "Lead"]
    salary_min: int
    salary_max: int
    industry: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    description: str
    posted_days_ago: int = 1

    def to_embedding_text(self) -> str:
        """Return rich text representation used for semantic embeddings."""
        return (
            f"{self.title} at {self.company} ({self.industry}, {self.work_mode}, {self.experience_level}). "
            f"Required skills: {', '.join(self.required_skills)}. "
            f"Nice to have: {', '.join(self.nice_to_have_skills)}. "
            f"Description: {self.description}"
        )


class JobMatch(BaseModel):
    """Ranked job recommendation with score breakdown and skill gap analysis."""

    job: JobPosting
    overall_score: float = Field(ge=0.0, le=100.0)
    semantic_score: float = Field(ge=0.0, le=100.0)
    skill_score: float = Field(ge=0.0, le=100.0)
    preference_score: float = Field(ge=0.0, le=100.0)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_nice_to_have: list[str] = Field(default_factory=list)
    fit_reasons: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Structured response returned by the CareerPilot AI Agent."""

    reply: str
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    referenced_jobs: list[JobMatch] = Field(default_factory=list)
    preferences_updated: bool = False
    mode: Literal["openai", "fallback"] = "fallback"
