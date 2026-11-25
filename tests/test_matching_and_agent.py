"""Tests for semantic search indexing, ranking, skill gaps, and AI agent tool calling."""

from __future__ import annotations

import json

import pytest

from careerpilot.agent import CareerPilotAgent, search_jobs
from careerpilot.data_loader import load_jobs, load_sample_resumes
from careerpilot.embeddings import SemanticJobIndex
from careerpilot.matcher import compute_profile_readiness, normalize_skill, rank_jobs
from careerpilot.models import UserPreferences
from careerpilot.resume_parser import extract_resume_fallback


@pytest.fixture(scope="module")
def tfidf_index() -> SemanticJobIndex:
    """Fast deterministic index without downloading HuggingFace weights during unit tests."""
    jobs = load_jobs()
    return SemanticJobIndex(jobs=jobs, prefer_transformer=False)


def test_skill_normalization_aliases() -> None:
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("Postgres") == "postgresql"
    assert normalize_skill("  Vector DB ") == "vector databases"


def test_semantic_ranking_surfaces_relevant_ai_roles(tfidf_index: SemanticJobIndex) -> None:
    samples = load_sample_resumes()
    alex_text = next(text for text in samples.values() if "Alex Rivera" in text)
    profile = extract_resume_fallback(alex_text)

    matches = rank_jobs(
        index=tfidf_index,
        profile=profile,
        preferences=UserPreferences(work_modes=["Remote"]),
        top_k=5,
    )
    assert len(matches) == 5
    top_titles = [m.job.title.lower() for m in matches]
    assert any("llm" in t or "ai" in t or "rag" in t for t in top_titles)
    # Scores should be sorted descending
    scores = [m.overall_score for m in matches]
    assert scores == sorted(scores, reverse=True)
    # Matched & missing skills populated
    assert any(len(m.matched_skills) > 0 for m in matches)


def test_strict_preference_filtering(tfidf_index: SemanticJobIndex) -> None:
    prefs = UserPreferences(work_modes=["Remote"], min_salary=160000)
    matches = rank_jobs(
        index=tfidf_index,
        preferences=prefs,
        query="Python AI Engineer",
        top_k=10,
        strict_preferences=True,
    )
    assert len(matches) > 0
    assert all(m.job.work_mode == "Remote" for m in matches)
    assert all(m.job.salary_max >= 160000 for m in matches)


def test_profile_readiness_summary(tfidf_index: SemanticJobIndex) -> None:
    samples = load_sample_resumes()
    alex_text = next(text for text in samples.values() if "Alex Rivera" in text)
    profile = extract_resume_fallback(alex_text)
    matches = rank_jobs(index=tfidf_index, profile=profile, top_k=10)
    readiness = compute_profile_readiness(profile, matches)

    assert readiness["completeness_score"] >= 80
    assert readiness["avg_top_fit"] > 0


def test_fallback_agent_search_and_preference_memory(tfidf_index: SemanticJobIndex) -> None:
    samples = load_sample_resumes()
    alex_text = next(text for text in samples.values() if "Alex Rivera" in text)
    profile = extract_resume_fallback(alex_text)
    prefs = UserPreferences()

    agent = CareerPilotAgent(index=tfidf_index, api_key=None)
    resp = agent.chat(
        user_message="Remember I prefer Remote roles paying at least $150k for LLM Engineer",
        history=[],
        profile=profile,
        preferences=prefs,
    )

    assert resp.mode == "fallback"
    assert resp.preferences_updated is True
    assert prefs.work_modes == ["Remote"]
    assert prefs.min_salary == 150000
    assert any(tc["tool"] == "save_user_preference" for tc in resp.tool_calls_made)
    assert any(tc["tool"] == "search_jobs" for tc in resp.tool_calls_made)
    assert len(resp.referenced_jobs) > 0


def test_fallback_agent_job_details_and_skill_gap(tfidf_index: SemanticJobIndex) -> None:
    prefs = UserPreferences()
    agent = CareerPilotAgent(index=tfidf_index, api_key=None)

    detail_resp = agent.chat("Tell me about JOB-001", history=[], profile=None, preferences=prefs)
    assert "JOB-001" in detail_resp.reply
    assert any(tc["tool"] == "get_job_details" for tc in detail_resp.tool_calls_made)

    gap_resp = agent.chat(
        "What skills am I missing for Senior AI roles?",
        history=[],
        profile=None,
        preferences=prefs,
    )
    assert any(tc["tool"] == "analyze_skill_gap" for tc in gap_resp.tool_calls_made)


def test_openai_agent_tool_calling_loop_mocked(
    tfidf_index: SemanticJobIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    class _FakeToolCall:
        id = "call_123"

        class _Fn:
            name = "search_jobs"
            arguments = json.dumps({"query": "RAG Engineer", "work_mode": "Remote", "top_k": 3})

        function = _Fn()

    class _FakeResponse1:
        class _Choice:
            class _Msg:
                content = ""
                tool_calls = [_FakeToolCall()]

            message = _Msg()

        choices = [_Choice()]

    class _FakeResponse2:
        class _Choice:
            class _Msg:
                content = "Here are the top 3 Remote RAG roles from search_jobs."
                tool_calls = None

            message = _Msg()

        choices = [_Choice()]

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return _FakeResponse1() if calls["count"] == 1 else _FakeResponse2()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.chat = _FakeChat()

    monkeypatch.setattr("careerpilot.agent.OpenAI", _FakeOpenAI)
    agent = CareerPilotAgent(index=tfidf_index, api_key="sk-fake")
    resp = agent.chat("Find remote RAG jobs", history=[], profile=None, preferences=UserPreferences())

    assert resp.mode == "openai"
    assert "top 3 Remote RAG roles" in resp.reply
    assert resp.tool_calls_made == [
        {"tool": "search_jobs", "arguments": {"query": "RAG Engineer", "work_mode": "Remote", "top_k": 3}}
    ]
    assert len(resp.referenced_jobs) == 3
    assert all(m.job.work_mode == "Remote" for m in resp.referenced_jobs)


def test_search_jobs_helper_returns_bounded_results(tfidf_index: SemanticJobIndex) -> None:
    results = search_jobs(
        index=tfidf_index,
        profile=None,
        preferences=UserPreferences(),
        query="Python Backend FastAPI",
        top_k=4,
    )
    assert len(results) == 4
