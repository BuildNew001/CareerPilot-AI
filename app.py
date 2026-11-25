"""Streamlit entrypoint for CareerPilot AI — Intelligent Job Search & Career Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure `src` is importable when launched via `streamlit run app.py`
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from careerpilot.agent import CareerPilotAgent
from careerpilot.config import Settings
from careerpilot.data_loader import jobs_to_dataframe, load_jobs, load_sample_resumes
from careerpilot.embeddings import SemanticJobIndex
from careerpilot.matcher import compute_profile_readiness, rank_jobs
from careerpilot.models import JobPosting, ResumeProfile, UserPreferences
from careerpilot.resume_parser import analyze_resume, extract_text_from_pdf
from careerpilot.storage import clear_user_state, load_user_state, save_user_state
from careerpilot.ui.components import (
    render_hero,
    render_job_card,
    render_kpi_card,
    render_profile_card,
)
from careerpilot.ui.styles import inject_custom_css

st.set_page_config(
    page_title="CareerPilot AI — Intelligent Job Search & Career Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_jobs_catalog() -> list[JobPosting]:
    """Load and cache the curated job postings dataset."""
    return load_jobs()


@st.cache_resource(show_spinner=False)
def get_semantic_index(embedding_model: str, prefer_transformer: bool = True) -> SemanticJobIndex:
    """Build and cache the semantic embedding index across reruns."""
    jobs = get_jobs_catalog()
    return SemanticJobIndex(
        jobs=jobs,
        model_name=embedding_model,
        prefer_transformer=prefer_transformer,
    )


def _init_session_state() -> None:
    """Initialize persisted preferences, profile, and chat history in session state."""
    if "initialized" in st.session_state:
        return
    saved_prefs, saved_profile = load_user_state()
    st.session_state.preferences = saved_prefs
    st.session_state.profile = saved_profile
    st.session_state.resume_text_input = saved_profile.raw_text if saved_profile else ""
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm **CareerPilot AI**. Upload or analyze your resume, then ask me to search roles, "
                "explain skill gaps, or remember your preferences (e.g., *'I prefer Remote roles paying $150k+'*)."
            ),
            "tool_calls": [],
            "jobs": [],
        }
    ]
    st.session_state.api_key_input = ""
    st.session_state.initialized = True


def main() -> None:
    inject_custom_css()
    _init_session_state()

    settings = Settings.from_env(api_key_override=st.session_state.get("api_key_input"))
    jobs = get_jobs_catalog()
    sample_resumes = load_sample_resumes()

    # Sidebar controls & persistent memory summary
    with st.sidebar:
        st.header("CareerPilot Controls")
        api_key_input = st.text_input(
            "OpenAI API Key (optional)",
            value=st.session_state.get("api_key_input", ""),
            type="password",
            help="Enables GPT structured resume parsing & OpenAI tool-calling agent. Leave blank for smart local demo mode.",
        )
        st.session_state.api_key_input = api_key_input
        settings = Settings.from_env(api_key_override=api_key_input)

        embedding_choice = st.radio(
            "Embedding Engine",
            options=["Sentence Transformers (MiniLM)", "Fast TF-IDF (Instant)"],
            index=0,
            help="MiniLM provides dense semantic embeddings; TF-IDF starts instantaneously offline.",
        )
        prefer_transformer = embedding_choice.startswith("Sentence")

    # Render loading indicator cleanly inside the page if model is initializing for the first time
    status_slot = st.empty()
    with (
        status_slot,
        st.spinner(
            "Loading semantic job index (first-time MiniLM download is ~90 MB)..."
            if prefer_transformer
            else "Building TF-IDF job index..."
        ),
    ):
        index = get_semantic_index(settings.embedding_model, prefer_transformer=prefer_transformer)
    status_slot.empty()

    with st.sidebar:

        st.divider()
        st.subheader("Quick Sample Resumes")
        if sample_resumes:
            selected_sample = st.selectbox(
                "Load a sample candidate",
                options=["-- Choose sample --", *sample_resumes.keys()],
            )
            if (
                st.button("Load & Analyze Sample", use_container_width=True)
                and selected_sample != "-- Choose sample --"
            ):
                sample_text = sample_resumes[selected_sample]
                st.session_state.resume_text_input = sample_text
                with st.spinner("Analyzing sample resume..."):
                    st.session_state.profile = analyze_resume(
                        sample_text,
                        api_key=settings.openai_api_key,
                        model=settings.openai_model,
                    )
                    save_user_state(st.session_state.preferences, st.session_state.profile)
                st.success(f"Loaded {selected_sample}")
                st.rerun()

        st.divider()
        st.subheader("Active Memory & Preferences")
        prefs: UserPreferences = st.session_state.preferences
        st.caption(f"Work Modes: {', '.join(prefs.work_modes) or 'Any'}")
        st.caption(f"Levels: {', '.join(prefs.experience_levels) or 'Any'}")
        st.caption(f"Min Salary: ${prefs.min_salary:,}" if prefs.min_salary else "Min Salary: Any")
        st.caption(f"Locations: {', '.join(prefs.preferred_locations) or 'Any'}")
        st.caption(f"Target Roles: {', '.join(prefs.preferred_roles) or 'Any'}")

        c_save, c_reset = st.columns(2)
        with c_save:
            if st.button("Save State", use_container_width=True):
                save_user_state(st.session_state.preferences, st.session_state.profile)
                st.toast("Preferences and profile saved locally.")
        with c_reset:
            if st.button("Reset", use_container_width=True):
                clear_user_state()
                st.session_state.preferences = UserPreferences()
                st.session_state.profile = None
                st.session_state.resume_text_input = ""
                st.rerun()

    render_hero(
        has_api_key=bool(settings.openai_api_key),
        embedding_backend=index.backend,
        job_count=len(jobs),
    )

    tab_resume, tab_jobs, tab_agent, tab_prefs = st.tabs(
        [
            "1. Resume Analyzer",
            "2. Job Discovery & Ranking",
            "3. AI Career Copilot",
            "4. Preferences & Dataset",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Resume Analyzer & Structured Extraction
    # ------------------------------------------------------------------
    with tab_resume:
        col_upload, col_paste = st.columns([1, 1.6])
        with col_upload:
            st.subheader("Upload Resume (PDF or TXT)")
            uploaded_file = st.file_uploader(
                "Drop your resume here",
                type=["pdf", "txt"],
                help="PDF text is extracted locally via PyPDF before analysis.",
            )
            if uploaded_file is not None:
                if uploaded_file.name.lower().endswith(".pdf"):
                    extracted = extract_text_from_pdf(uploaded_file.getvalue())
                else:
                    extracted = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                if extracted:
                    st.session_state.resume_text_input = extracted
                    st.success(f"Extracted {len(extracted)} characters from {uploaded_file.name}.")
                else:
                    st.warning("Could not extract readable text from that file.")

        with col_paste:
            st.subheader("Resume Content")
            resume_text = st.text_area(
                "Paste or review extracted resume text",
                value=st.session_state.get("resume_text_input", ""),
                height=230,
                placeholder="Paste your resume text here or load a sample candidate from the sidebar...",
            )
            st.session_state.resume_text_input = resume_text

            if st.button("Analyze Resume & Score Fit", type="primary"):
                if not resume_text.strip():
                    st.error("Please upload a PDF/TXT or paste resume text first.")
                else:
                    with st.spinner("Extracting skills, experience, and career signals..."):
                        profile = analyze_resume(
                            resume_text=resume_text,
                            api_key=settings.openai_api_key,
                            model=settings.openai_model,
                        )
                        st.session_state.profile = profile
                        save_user_state(st.session_state.preferences, profile)
                    st.success("Resume analyzed successfully!")

        st.divider()
        profile: ResumeProfile | None = st.session_state.profile
        if profile is None:
            st.info(
                "No resume analyzed yet. Upload a PDF, paste text above, or load a sample candidate from the sidebar."
            )
        else:
            top_for_readiness = rank_jobs(
                index=index,
                profile=profile,
                preferences=st.session_state.preferences,
                top_k=10,
            )
            readiness = compute_profile_readiness(profile, top_for_readiness)
            render_profile_card(profile, readiness)

    # ------------------------------------------------------------------
    # TAB 2: Semantic Job Discovery & Ranking
    # ------------------------------------------------------------------
    with tab_jobs:
        st.subheader("Semantic Job Search & Personalized Ranking")
        f1, f2, f3, f4 = st.columns([2.2, 1, 1, 1])
        with f1:
            search_query = st.text_input(
                "Search query (optional — combines with your resume)",
                placeholder="e.g., LLM agents, RAG, FastAPI, Remote",
            )
        with f2:
            work_modes = st.multiselect("Work Mode", ["Remote", "Hybrid", "Onsite"])
        with f3:
            levels = st.multiselect("Level", ["Entry", "Mid", "Senior", "Lead"])
        with f4:
            min_sal = st.number_input(
                "Min Salary ($)",
                min_value=0,
                max_value=300000,
                step=10000,
                value=0,
            )

        c_opt1, c_opt2, c_opt3 = st.columns([1.2, 1.2, 1])
        with c_opt1:
            top_k = st.slider("Results to show", min_value=5, max_value=30, value=10, step=5)
        with c_opt2:
            strict_prefs = st.checkbox(
                "Strictly filter by saved preferences",
                value=False,
                help="When unchecked, saved preferences boost rankings softly instead of hiding roles.",
            )
        with c_opt3:
            view_mode = st.radio("View", ["Cards", "Table"], horizontal=True)

        ranked = rank_jobs(
            index=index,
            profile=st.session_state.profile,
            preferences=st.session_state.preferences,
            query=search_query,
            top_k=top_k,
            work_mode_filter=work_modes or None,
            level_filter=levels or None,
            min_salary_filter=int(min_sal) if min_sal > 0 else None,
            strict_preferences=strict_prefs,
        )

        if ranked:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("Matches Shown", str(len(ranked)), f"Out of {len(jobs)} roles")
            with k2:
                render_kpi_card("Top Fit Score", f"{ranked[0].overall_score:.0f}%", ranked[0].job.title)
            with k3:
                avg_sem = sum(m.semantic_score for m in ranked) / len(ranked)
                render_kpi_card("Avg Semantic Match", f"{avg_sem:.0f}%", "Cosine similarity")
            with k4:
                remote_cnt = sum(1 for m in ranked if m.job.work_mode == "Remote")
                render_kpi_card("Remote Roles", str(remote_cnt), "In current shortlist")
            st.markdown("")

            if view_mode == "Cards":
                for idx_num, match in enumerate(ranked, start=1):
                    render_job_card(match, rank=idx_num)
            else:
                table_df = pd.DataFrame(
                    [
                        {
                            "Rank": i,
                            "Job ID": m.job.job_id,
                            "Title": m.job.title,
                            "Company": m.job.company,
                            "Mode": m.job.work_mode,
                            "Level": m.job.experience_level,
                            "Salary Min": m.job.salary_min,
                            "Salary Max": m.job.salary_max,
                            "Fit %": m.overall_score,
                            "Semantic %": m.semantic_score,
                            "Skill %": m.skill_score,
                            "Matched Skills": ", ".join(m.matched_skills),
                            "Missing Skills": ", ".join(m.missing_skills),
                        }
                        for i, m in enumerate(ranked, start=1)
                    ]
                )
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Ranked Matches CSV",
                    data=table_df.to_csv(index=False).encode("utf-8"),
                    file_name="careerpilot_ranked_matches.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No jobs matched your active filters. Try relaxing the filters.")

    # ------------------------------------------------------------------
    # TAB 3: AI Career Copilot (Tool-Calling Agent)
    # ------------------------------------------------------------------
    with tab_agent:
        st.subheader("AI Career Copilot (Tool-Calling Agent)")
        st.caption(
            "Ask career questions, search the job catalog via `search_jobs()`, inspect specific `JOB-XXX` listings, "
            "or tell the agent your preferences to save them into memory."
        )

        quick_prompts = [
            "Find top Remote AI Engineer roles paying at least $150k",
            "What skills am I missing for Senior LLM roles?",
            "Remember I prefer Remote roles and Senior level",
            "Tell me more about JOB-001",
        ]
        q_cols = st.columns(len(quick_prompts))
        queued_prompt: str | None = None
        for col, qp in zip(q_cols, quick_prompts, strict=True):
            with col:
                if st.button(qp, use_container_width=True):
                    queued_prompt = qp

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    with st.expander(f"Tool Calls Executed ({len(msg['tool_calls'])})"):
                        st.json(msg["tool_calls"])
                if msg.get("jobs"):
                    for job_match in msg["jobs"][:3]:
                        render_job_card(job_match)

        user_prompt = st.chat_input("Ask CareerPilot AI about jobs, skills, or career strategy...")
        active_prompt = queued_prompt or user_prompt
        if active_prompt:
            st.session_state.chat_history.append(
                {"role": "user", "content": active_prompt, "tool_calls": [], "jobs": []}
            )
            agent = CareerPilotAgent(
                index=index,
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]
            ]
            with st.spinner("CareerPilot Agent is reasoning and calling tools..."):
                response = agent.chat(
                    user_message=active_prompt,
                    history=history_payload,
                    profile=st.session_state.profile,
                    preferences=st.session_state.preferences,
                )

            if response.preferences_updated:
                save_user_state(st.session_state.preferences, st.session_state.profile)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.reply,
                    "tool_calls": response.tool_calls_made,
                    "jobs": response.referenced_jobs,
                }
            )
            st.rerun()

    # ------------------------------------------------------------------
    # TAB 4: Preferences & Dataset Explorer
    # ------------------------------------------------------------------
    with tab_prefs:
        col_p, col_d = st.columns([1, 1.35])
        with col_p:
            st.subheader("Edit Saved Preferences")
            with st.form("preferences_form"):
                current_prefs: UserPreferences = st.session_state.preferences
                pref_roles = st.text_input(
                    "Preferred Roles (comma-separated)",
                    value=", ".join(current_prefs.preferred_roles),
                    placeholder="AI Engineer, ML Engineer, Applied Scientist",
                )
                pref_locs = st.text_input(
                    "Preferred Locations (comma-separated)",
                    value=", ".join(current_prefs.preferred_locations),
                    placeholder="San Francisco, New York, Remote",
                )
                pref_modes = st.multiselect(
                    "Preferred Work Modes",
                    ["Remote", "Hybrid", "Onsite"],
                    default=list(current_prefs.work_modes),
                )
                pref_levels = st.multiselect(
                    "Target Experience Levels",
                    ["Entry", "Mid", "Senior", "Lead"],
                    default=list(current_prefs.experience_levels),
                )
                pref_salary = st.number_input(
                    "Target Minimum Salary ($)",
                    min_value=0,
                    max_value=400000,
                    step=5000,
                    value=int(current_prefs.min_salary or 0),
                )
                pref_industries = st.text_input(
                    "Preferred Industries (comma-separated)",
                    value=", ".join(current_prefs.preferred_industries),
                    placeholder="Artificial Intelligence, FinTech, Developer Tools",
                )
                submitted = st.form_submit_button("Save Preferences", type="primary")
                if submitted:
                    updated = UserPreferences(
                        preferred_roles=[x.strip() for x in pref_roles.split(",") if x.strip()],
                        preferred_locations=[x.strip() for x in pref_locs.split(",") if x.strip()],
                        work_modes=pref_modes,
                        min_salary=int(pref_salary) if pref_salary > 0 else None,
                        experience_levels=pref_levels,
                        preferred_industries=[
                            x.strip() for x in pref_industries.split(",") if x.strip()
                        ],
                        notes=current_prefs.notes,
                    )
                    st.session_state.preferences = updated
                    save_user_state(updated, st.session_state.profile)
                    st.success("Preferences updated and saved!")
                    st.rerun()

            if current_prefs.notes:
                st.markdown("**Agent Memory Notes**")
                for note in current_prefs.notes:
                    st.markdown(f"- {note}")

        with col_d:
            st.subheader(f"Curated Job Dataset ({len(jobs)} Roles)")
            df = jobs_to_dataframe(jobs)
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.caption("Roles by Work Mode")
                st.bar_chart(df["work_mode"].value_counts())
            with c_chart2:
                st.caption("Roles by Experience Level")
                st.bar_chart(df["experience_level"].value_counts())
            st.dataframe(df, use_container_width=True, height=340, hide_index=True)


if __name__ == "__main__":
    main()
