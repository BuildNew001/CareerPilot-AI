"""Tests for dataset loading, resume parsing, and storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from careerpilot.data_loader import jobs_to_dataframe, load_jobs, load_sample_resumes
from careerpilot.models import ResumeProfile, UserPreferences
from careerpilot.resume_parser import analyze_resume, extract_resume_fallback
from careerpilot.storage import clear_user_state, load_user_state, save_user_state


def test_load_jobs_catalog_has_80_valid_entries() -> None:
    jobs = load_jobs()
    assert len(jobs) == 80
    assert all(j.job_id.startswith("JOB-") for j in jobs)
    assert all(j.salary_max >= j.salary_min > 0 for j in jobs)
    assert all(len(j.required_skills) >= 3 for j in jobs)

    df = jobs_to_dataframe(jobs)
    assert len(df) == 80
    assert "title" in df.columns


def test_load_sample_resumes() -> None:
    samples = load_sample_resumes()
    assert len(samples) >= 3
    assert any("Alex Rivera" in text for text in samples.values())


def test_fallback_resume_extraction() -> None:
    samples = load_sample_resumes()
    alex_text = next(text for text in samples.values() if "Alex Rivera" in text)
    profile = extract_resume_fallback(alex_text)

    assert profile.candidate_name == "Alex Rivera"
    assert profile.years_of_experience >= 5.0
    assert profile.seniority_level == "Senior"
    assert "Python" in profile.all_skills()
    assert "OpenAI API" in profile.all_skills()
    assert "Sentence Transformers" in profile.all_skills()
    assert profile.extraction_mode == "fallback"


def test_analyze_resume_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        analyze_resume("   ")


def test_analyze_resume_openai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeParsed:
        candidate_name = "Test Candidate"
        current_title = "LLM Engineer"
        years_of_experience = 4.0
        seniority_level = "Senior"
        skills = ["Python", "RAG"]
        tools_and_frameworks = ["OpenAI API", "FastAPI"]
        domains = ["AI"]
        education = ["M.S. CS"]
        career_interests = ["AI Agents"]
        summary = "Experienced LLM builder."

    class _FakeChoice:
        class _Msg:
            parsed = _FakeParsed()

        message = _Msg()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        @staticmethod
        def parse(**kwargs):  # type: ignore[no-untyped-def]
            return _FakeCompletion()

    class _FakeBeta:
        class _Chat:
            completions = _FakeCompletions()

        chat = _Chat()

    class _FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.beta = _FakeBeta()

    monkeypatch.setattr("careerpilot.resume_parser.OpenAI", _FakeOpenAI)
    profile = analyze_resume("Experienced Python & RAG engineer", api_key="sk-test-key")
    assert profile.extraction_mode == "openai"
    assert profile.candidate_name == "Test Candidate"
    assert "RAG" in profile.all_skills()


def test_storage_roundtrip(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    prefs = UserPreferences(
        preferred_roles=["AI Engineer"],
        work_modes=["Remote"],
        min_salary=150000,
        notes=["Prefer developer tools"],
    )
    profile = ResumeProfile(candidate_name="Alex", skills=["Python", "FastAPI"])

    save_user_state(prefs, profile, state_path=state_file)
    loaded_prefs, loaded_profile = load_user_state(state_path=state_file)

    assert loaded_prefs.work_modes == ["Remote"]
    assert loaded_prefs.min_salary == 150000
    assert loaded_profile is not None
    assert loaded_profile.candidate_name == "Alex"

    clear_user_state(state_path=state_file)
    empty_prefs, empty_profile = load_user_state(state_path=state_file)
    assert empty_prefs.work_modes == []
    assert empty_profile is None
