"""Job ranking, skill gap analysis, and preference-aware personalization."""

from __future__ import annotations

from careerpilot.embeddings import SemanticJobIndex
from careerpilot.models import JobMatch, JobPosting, ResumeProfile, UserPreferences

# Canonical aliases so equivalent skill labels match cleanly
SKILL_ALIASES: dict[str, str] = {
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "openai": "openai api",
    "gpt": "openai api",
    "llms": "llm architecture",
    "llm": "llm architecture",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "hf": "hugging face",
    "huggingface": "hugging face",
    "spark": "apache spark",
    "pyspark": "apache spark",
    "vector db": "vector databases",
    "vector database": "vector databases",
    "ab testing": "a/b testing",
    "a/b test": "a/b testing",
    "rest api": "rest apis",
    "node": "node.js",
    "nodejs": "node.js",
    "next": "next.js",
    "nextjs": "next.js",
    "tool calling": "function calling",
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill name for reliable set comparison."""
    cleaned = " ".join(skill.strip().lower().split())
    return SKILL_ALIASES.get(cleaned, cleaned)


def _compute_skill_overlap(
    candidate_skills: list[str],
    required_skills: list[str],
    nice_to_have: list[str],
) -> tuple[float, list[str], list[str], list[str]]:
    """Return (skill_score_0_100, matched_req, missing_req, matched_nice)."""
    cand_norm = {normalize_skill(s) for s in candidate_skills if s.strip()}

    matched_req: list[str] = []
    missing_req: list[str] = []
    for req in required_skills:
        if normalize_skill(req) in cand_norm:
            matched_req.append(req)
        else:
            missing_req.append(req)

    matched_nice = [s for s in nice_to_have if normalize_skill(s) in cand_norm]

    req_ratio = (len(matched_req) / len(required_skills)) if required_skills else 0.5
    nice_bonus = (0.12 * (len(matched_nice) / len(nice_to_have))) if nice_to_have else 0.0
    score = min(100.0, (req_ratio * 100.0) + (nice_bonus * 100.0))
    return round(score, 1), matched_req, missing_req, matched_nice


def _compute_preference_score(
    job: JobPosting,
    preferences: UserPreferences,
) -> tuple[float, list[str]]:
    """Score how well a job aligns with stored user preferences (0-100)."""
    checks: list[float] = []
    reasons: list[str] = []

    if preferences.work_modes:
        if job.work_mode in preferences.work_modes:
            checks.append(1.0)
            reasons.append(f"Matches preferred {job.work_mode} work mode")
        else:
            checks.append(0.0)

    if preferences.min_salary:
        if job.salary_max >= preferences.min_salary:
            checks.append(1.0)
            if job.salary_min >= preferences.min_salary:
                reasons.append(f"Meets salary target (>= ${preferences.min_salary:,})")
        else:
            checks.append(0.15)

    if preferences.preferred_locations:
        loc_lower = job.location.lower()
        if any(loc.lower() in loc_lower for loc in preferences.preferred_locations) or (
            "remote" in [loc.lower() for loc in preferences.preferred_locations]
            and job.work_mode == "Remote"
        ):
            checks.append(1.0)
            reasons.append(f"Matches location preference ({job.location})")
        else:
            checks.append(0.2)

    if preferences.experience_levels:
        if job.experience_level in preferences.experience_levels:
            checks.append(1.0)
            reasons.append(f"Matches {job.experience_level} seniority")
        else:
            checks.append(0.25)

    if preferences.preferred_roles:
        title_lower = job.title.lower()
        if any(role.lower() in title_lower for role in preferences.preferred_roles):
            checks.append(1.0)
            reasons.append(f"Aligns with target role ({job.title})")
        else:
            checks.append(0.35)

    if preferences.preferred_industries:
        ind_lower = job.industry.lower()
        if any(ind.lower() in ind_lower for ind in preferences.preferred_industries):
            checks.append(1.0)
            reasons.append(f"In preferred industry ({job.industry})")
        else:
            checks.append(0.3)

    if not checks:
        return 75.0, reasons

    return round((sum(checks) / len(checks)) * 100.0, 1), reasons


def rank_jobs(
    index: SemanticJobIndex,
    profile: ResumeProfile | None = None,
    preferences: UserPreferences | None = None,
    query: str | None = None,
    top_k: int = 15,
    work_mode_filter: list[str] | None = None,
    level_filter: list[str] | None = None,
    min_salary_filter: int | None = None,
    strict_preferences: bool = False,
) -> list[JobMatch]:
    """Rank jobs by semantic similarity, skill match, and user preferences."""
    prefs = preferences or UserPreferences()

    query_parts: list[str] = []
    if query and query.strip():
        query_parts.append(query.strip())
    if profile is not None:
        query_parts.append(profile.to_search_query())
    if prefs.preferred_roles:
        query_parts.append("Target roles: " + ", ".join(prefs.preferred_roles))
    if prefs.preferred_industries:
        query_parts.append("Industries: " + ", ".join(prefs.preferred_industries))

    combined_query = " | ".join(query_parts) or "Software AI Data Engineering"
    semantic_sims = index.score_query(combined_query)

    cand_skills = profile.all_skills() if profile is not None else []
    matches: list[JobMatch] = []

    for job, sem_sim in zip(index.jobs, semantic_sims, strict=True):
        if work_mode_filter and job.work_mode not in work_mode_filter:
            continue
        if level_filter and job.experience_level not in level_filter:
            continue
        if min_salary_filter and job.salary_max < min_salary_filter:
            continue

        if strict_preferences:
            if prefs.work_modes and job.work_mode not in prefs.work_modes:
                continue
            if prefs.min_salary and job.salary_max < prefs.min_salary:
                continue
            if prefs.experience_levels and job.experience_level not in prefs.experience_levels:
                continue

        # Normalize semantic similarity nicely into 0-100 scale
        # Typical MiniLM cosine similarity for strong job matches sits in 0.35 - 0.78
        sem_pct = round(min(100.0, max(0.0, sem_sim * 135.0)), 1)

        if cand_skills:
            skill_pct, matched_req, missing_req, matched_nice = _compute_skill_overlap(
                cand_skills, job.required_skills, job.nice_to_have_skills
            )
        else:
            skill_pct = 50.0
            matched_req = []
            missing_req = list(job.required_skills)
            matched_nice = []

        pref_pct, pref_reasons = _compute_preference_score(job, prefs)

        if cand_skills:
            overall = round((0.50 * sem_pct) + (0.35 * skill_pct) + (0.15 * pref_pct), 1)
        else:
            overall = round((0.75 * sem_pct) + (0.25 * pref_pct), 1)

        fit_reasons: list[str] = []
        if matched_req:
            fit_reasons.append(
                f"Matches {len(matched_req)}/{len(job.required_skills)} required skills ({', '.join(matched_req[:4])})"
            )
        if sem_pct >= 65.0:
            fit_reasons.append("Strong semantic alignment with your experience and goals")
        fit_reasons.extend(pref_reasons)

        matches.append(
            JobMatch(
                job=job,
                overall_score=min(100.0, overall),
                semantic_score=sem_pct,
                skill_score=skill_pct,
                preference_score=pref_pct,
                matched_skills=matched_req,
                missing_skills=missing_req,
                matched_nice_to_have=matched_nice,
                fit_reasons=fit_reasons,
            )
        )

    matches.sort(key=lambda m: (m.overall_score, m.semantic_score, m.skill_score), reverse=True)
    return matches[:top_k]


def compute_profile_readiness(profile: ResumeProfile, top_matches: list[JobMatch]) -> dict[str, object]:
    """Compute ATS-style profile readiness summary and top skill gaps across top matches."""
    skill_count = len(profile.all_skills())
    completeness = 0
    if profile.candidate_name and profile.candidate_name != "Candidate":
        completeness += 15
    if profile.current_title:
        completeness += 15
    if profile.summary:
        completeness += 20
    if skill_count >= 5:
        completeness += 25
    elif skill_count > 0:
        completeness += 12
    if profile.education:
        completeness += 10
    if profile.domains or profile.career_interests:
        completeness += 15

    avg_top_fit = (
        round(sum(m.overall_score for m in top_matches[:5]) / min(5, len(top_matches)), 1)
        if top_matches
        else 0.0
    )

    gap_counts: dict[str, int] = {}
    for m in top_matches[:10]:
        for skill in m.missing_skills:
            gap_counts[skill] = gap_counts.get(skill, 0) + 1

    top_gaps = sorted(gap_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
    return {
        "completeness_score": min(100, completeness),
        "avg_top_fit": avg_top_fit,
        "top_skill_gaps": top_gaps,
    }
