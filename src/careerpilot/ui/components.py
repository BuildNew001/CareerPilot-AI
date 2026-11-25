"""Reusable UI rendering components for CareerPilot AI."""

from __future__ import annotations

import html

import streamlit as st

from careerpilot.models import JobMatch, ResumeProfile


def _esc(text: object) -> str:
    return html.escape(str(text))


def render_hero(has_api_key: bool, embedding_backend: str, job_count: int) -> None:
    """Render top application banner with live runtime mode indicators."""
    llm_badge = (
        '<span class="cp-status-pill live">LLM Mode: OpenAI API Connected</span>'
        if has_api_key
        else '<span class="cp-status-pill">LLM Mode: Local Smart Demo (Add API Key for OpenAI)</span>'
    )
    st.markdown(
        f"""
        <div class="cp-hero">
            <h1>CareerPilot AI — Intelligent Job Search &amp; Career Agent</h1>
            <p>
                Structured LLM resume intelligence, semantic Sentence Transformer retrieval,
                skill-gap scoring, and tool-calling career guidance.
            </p>
            <div class="cp-badge-row">
                {llm_badge}
                <span class="cp-status-pill">Embeddings: {_esc(embedding_backend)}</span>
                <span class="cp-status-pill">Curated Roles: {job_count}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str = "") -> None:
    """Render a compact KPI metric card."""
    st.markdown(
        f"""
        <div class="cp-kpi">
            <div class="cp-kpi-label">{_esc(label)}</div>
            <div class="cp-kpi-value">{_esc(value)}</div>
            <div class="cp-kpi-sub">{_esc(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_pills(items: list[str], variant: str = "neutral") -> str:
    """Build safe HTML for pill badges."""
    if not items:
        return '<span class="cp-pill cp-pill-neutral">None</span>'
    css_cls = {
        "match": "cp-pill-match",
        "miss": "cp-pill-miss",
        "indigo": "cp-pill-indigo",
        "neutral": "cp-pill-neutral",
    }.get(variant, "cp-pill-neutral")
    return "".join(f'<span class="cp-pill {css_cls}">{_esc(item)}</span>' for item in items)


def render_profile_card(profile: ResumeProfile, readiness: dict[str, object]) -> None:
    """Render structured candidate analysis and ATS readiness summary."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Candidate", profile.candidate_name, profile.current_title)
    with c2:
        render_kpi_card(
            "Experience",
            f"{profile.years_of_experience:g} yrs",
            f"Tier: {profile.seniority_level}",
        )
    with c3:
        render_kpi_card(
            "ATS Readiness",
            f"{readiness['completeness_score']}%",
            f"Extractor: {profile.extraction_mode.upper()}",
        )
    with c4:
        render_kpi_card(
            "Avg Top-5 Fit",
            f"{readiness['avg_top_fit']}%",
            f"{len(profile.all_skills())} skills detected",
        )

    st.markdown("")
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Extracted Profile & Skills")
        st.write(profile.summary)
        st.markdown("**Core Skills**")
        st.markdown(render_skill_pills(profile.skills, "indigo"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Tools, Frameworks & Cloud**")
        st.markdown(
            render_skill_pills(profile.tools_and_frameworks, "match"),
            unsafe_allow_html=True,
        )

    with col_right:
        st.subheader("Career Signal & Top Skill Gaps")
        if profile.domains:
            st.markdown("**Domains:** " + ", ".join(profile.domains))
        if profile.career_interests:
            st.markdown("**Target Interests:** " + ", ".join(profile.career_interests))
        if profile.education:
            st.markdown("**Education:**")
            for edu in profile.education:
                st.markdown(f"- {edu}")

        st.markdown("**Most Requested Missing Skills (Top 10 Matches)**")
        top_gaps = readiness.get("top_skill_gaps") or []
        if top_gaps:
            gap_labels = [f"{skill} ({count} roles)" for skill, count in top_gaps]
            st.markdown(render_skill_pills(gap_labels, "miss"), unsafe_allow_html=True)
        else:
            st.success("Strong coverage across top matching roles!")


def render_job_card(match: JobMatch, rank: int | None = None) -> None:
    """Render an interactive job match card with fit breakdown and skill pills."""
    job = match.job
    score = match.overall_score
    score_cls = "cp-score-high" if score >= 70 else ("cp-score-mid" if score >= 50 else "cp-score-low")
    prefix = f"#{rank} · " if rank is not None else ""

    matched_html = render_skill_pills(match.matched_skills, "match")
    missing_html = render_skill_pills(match.missing_skills, "miss")

    st.markdown(
        f"""
        <div class="cp-job-card">
            <div class="cp-job-header">
                <div>
                    <div class="cp-job-title">{_esc(prefix)}{_esc(job.title)}</div>
                    <div class="cp-job-meta">
                        <strong>{_esc(job.company)}</strong> · {_esc(job.location)} ·
                        {_esc(job.work_mode)} · {_esc(job.experience_level)} ·
                        <strong>${job.salary_min:,} - ${job.salary_max:,}</strong> ·
                        <code>{_esc(job.job_id)}</code>
                    </div>
                </div>
                <div class="cp-score-box {score_cls}">
                    <div style="font-size: 1.25rem;">{score:.0f}%</div>
                    <div style="font-size: 0.68rem; text-transform: uppercase;">Fit Score</div>
                </div>
            </div>
            <div style="margin-top: 0.7rem; font-size: 0.82rem; color: #475569;">
                Semantic: <strong>{match.semantic_score:.0f}%</strong> ·
                Skill Overlap: <strong>{match.skill_score:.0f}%</strong> ·
                Preference Fit: <strong>{match.preference_score:.0f}%</strong> ·
                Industry: <strong>{_esc(job.industry)}</strong>
            </div>
            <div style="margin-top: 0.65rem;">
                <span style="font-size: 0.78rem; font-weight: 600; color: #065F46;">Matched: </span>
                {matched_html}
            </div>
            <div style="margin-top: 0.35rem;">
                <span style="font-size: 0.78rem; font-weight: 600; color: #991B1B;">Missing: </span>
                {missing_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"View description & details — {job.job_id} ({job.title})"):
        st.write(job.description)
        if match.matched_nice_to_have:
            st.markdown(f"**Matched Bonus Skills:** {', '.join(match.matched_nice_to_have)}")
        elif job.nice_to_have_skills:
            st.markdown(f"**Nice-to-Have Skills:** {', '.join(job.nice_to_have_skills)}")
        if match.fit_reasons:
            st.markdown("**Why this fits:**")
            for reason in match.fit_reasons:
                st.markdown(f"- {reason}")
        st.caption(f"Posted {job.posted_days_ago} day(s) ago")
