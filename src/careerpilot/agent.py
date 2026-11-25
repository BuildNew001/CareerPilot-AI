"""AI Career Assistant with OpenAI function/tool calling and offline demo fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from careerpilot.embeddings import SemanticJobIndex
from careerpilot.matcher import rank_jobs
from careerpilot.models import AgentResponse, JobMatch, ResumeProfile, UserPreferences

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search and rank jobs from the CareerPilot job dataset using semantic embeddings, "
                "candidate skills, and optional filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Role title, skills, or career focus to search for.",
                    },
                    "work_mode": {
                        "type": "string",
                        "enum": ["Remote", "Hybrid", "Onsite"],
                        "description": "Optional work mode filter.",
                    },
                    "experience_level": {
                        "type": "string",
                        "enum": ["Entry", "Mid", "Senior", "Lead"],
                        "description": "Optional seniority level filter.",
                    },
                    "min_salary": {
                        "type": "integer",
                        "description": "Optional minimum salary threshold in USD.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top matching jobs to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Retrieve full job description and skill gap breakdown for a specific job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job identifier such as JOB-001.",
                    }
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_skill_gap",
            "description": (
                "Analyze matched vs missing skills for the candidate against either a specific job_id "
                "or a target role query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Either a job_id (e.g. JOB-001) or target role (e.g. 'LLM Engineer').",
                    }
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_preference",
            "description": (
                "Save or update the candidate's job search preferences in memory whenever they express "
                "location, remote/hybrid, salary, target role, seniority, or industry preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target role titles to remember.",
                    },
                    "preferred_locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred cities, countries, or 'Remote'.",
                    },
                    "work_modes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["Remote", "Hybrid", "Onsite"]},
                        "description": "Preferred work modes.",
                    },
                    "min_salary": {
                        "type": "integer",
                        "description": "Minimum target annual compensation in USD.",
                    },
                    "experience_levels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["Entry", "Mid", "Senior", "Lead"]},
                        "description": "Preferred experience levels.",
                    },
                    "preferred_industries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred industries.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Short career preference note to remember.",
                    },
                },
            },
        },
    },
]


def search_jobs(
    index: SemanticJobIndex,
    profile: ResumeProfile | None,
    preferences: UserPreferences,
    query: str,
    work_mode: str | None = None,
    experience_level: str | None = None,
    min_salary: int | None = None,
    top_k: int = 5,
) -> list[JobMatch]:
    """Execute semantic + skill + preference job search (also exposed as an AI agent tool)."""
    k = max(1, min(10, int(top_k)))
    return rank_jobs(
        index=index,
        profile=profile,
        preferences=preferences,
        query=query,
        top_k=k,
        work_mode_filter=[work_mode] if work_mode else None,
        level_filter=[experience_level] if experience_level else None,
        min_salary_filter=min_salary,
    )


class CareerPilotAgent:
    """Conversational AI Career Assistant backed by OpenAI tool calling."""

    def __init__(
        self,
        index: SemanticJobIndex,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.index = index
        self.api_key = api_key
        self.model = model
        self._jobs_by_id = {job.job_id.upper(): job for job in index.jobs}

    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        profile: ResumeProfile | None,
        preferences: UserPreferences,
        collected_matches: list[JobMatch],
    ) -> tuple[dict[str, Any], bool]:
        """Run a registered Python tool function and return JSON-serializable output."""
        prefs_changed = False

        if name == "search_jobs":
            matches = search_jobs(
                index=self.index,
                profile=profile,
                preferences=preferences,
                query=str(args.get("query", "")),
                work_mode=args.get("work_mode"),
                experience_level=args.get("experience_level"),
                min_salary=args.get("min_salary"),
                top_k=int(args.get("top_k", 5)),
            )
            collected_matches.extend(matches)
            return {
                "count": len(matches),
                "results": [
                    {
                        "job_id": m.job.job_id,
                        "title": m.job.title,
                        "company": m.job.company,
                        "location": m.job.location,
                        "work_mode": m.job.work_mode,
                        "experience_level": m.job.experience_level,
                        "salary_range": f"${m.job.salary_min:,}-${m.job.salary_max:,}",
                        "overall_score": m.overall_score,
                        "semantic_score": m.semantic_score,
                        "skill_score": m.skill_score,
                        "matched_skills": m.matched_skills,
                        "missing_skills": m.missing_skills,
                    }
                    for m in matches
                ],
            }, False

        if name == "get_job_details":
            job_id = str(args.get("job_id", "")).strip().upper()
            job = self._jobs_by_id.get(job_id)
            if job is None:
                return {"error": f"Job ID '{job_id}' not found."}, False

            # Score this single job relative to candidate
            scored = rank_jobs(
                index=self.index,
                profile=profile,
                preferences=preferences,
                query=job.title,
                top_k=len(self.index.jobs),
            )
            match_obj = next((m for m in scored if m.job.job_id == job.job_id), None)
            if match_obj:
                collected_matches.append(match_obj)
            return {
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "work_mode": job.work_mode,
                "experience_level": job.experience_level,
                "salary_range": f"${job.salary_min:,}-${job.salary_max:,}",
                "industry": job.industry,
                "required_skills": job.required_skills,
                "nice_to_have_skills": job.nice_to_have_skills,
                "description": job.description,
                "overall_score": match_obj.overall_score if match_obj else None,
                "matched_skills": match_obj.matched_skills if match_obj else [],
                "missing_skills": match_obj.missing_skills if match_obj else job.required_skills,
            }, False

        if name == "analyze_skill_gap":
            target = str(args.get("target", "")).strip()
            if target.upper() in self._jobs_by_id:
                job = self._jobs_by_id[target.upper()]
                scored = rank_jobs(
                    index=self.index,
                    profile=profile,
                    preferences=preferences,
                    query=job.title,
                    top_k=len(self.index.jobs),
                )
                target_matches = [m for m in scored if m.job.job_id == job.job_id]
            else:
                target_matches = search_jobs(
                    index=self.index,
                    profile=profile,
                    preferences=preferences,
                    query=target or "AI Engineer",
                    top_k=5,
                )

            collected_matches.extend(target_matches[:3])
            gap_freq: dict[str, int] = {}
            matched_set: set[str] = set()
            for m in target_matches:
                matched_set.update(m.matched_skills)
                for skill in m.missing_skills:
                    gap_freq[skill] = gap_freq.get(skill, 0) + 1
            ranked_gaps = [k for k, _ in sorted(gap_freq.items(), key=lambda kv: kv[1], reverse=True)]
            return {
                "target": target,
                "jobs_analyzed": [f"{m.job.job_id} {m.job.title} ({m.overall_score}%)" for m in target_matches],
                "candidate_matched_skills": sorted(matched_set),
                "priority_missing_skills": ranked_gaps[:8],
            }, False

        if name == "save_user_preference":
            if args.get("preferred_roles"):
                preferences.preferred_roles = list(dict.fromkeys(args["preferred_roles"]))
                prefs_changed = True
            if args.get("preferred_locations"):
                preferences.preferred_locations = list(dict.fromkeys(args["preferred_locations"]))
                prefs_changed = True
            if args.get("work_modes"):
                preferences.work_modes = list(dict.fromkeys(args["work_modes"]))
                prefs_changed = True
            if args.get("min_salary") is not None:
                preferences.min_salary = int(args["min_salary"])
                prefs_changed = True
            if args.get("experience_levels"):
                preferences.experience_levels = list(dict.fromkeys(args["experience_levels"]))
                prefs_changed = True
            if args.get("preferred_industries"):
                preferences.preferred_industries = list(dict.fromkeys(args["preferred_industries"]))
                prefs_changed = True
            if args.get("note"):
                note = str(args["note"]).strip()
                if note and note not in preferences.notes:
                    preferences.notes.append(note)
                    prefs_changed = True

            return {
                "status": "saved",
                "preferences": preferences.model_dump(),
            }, prefs_changed

        return {"error": f"Unknown tool: {name}"}, False

    def _build_system_prompt(
        self,
        profile: ResumeProfile | None,
        preferences: UserPreferences,
    ) -> str:
        profile_block = (
            profile.model_dump(exclude={"raw_text"}) if profile is not None else "No resume uploaded yet."
        )
        return (
            "You are CareerPilot AI, a pragmatic, encouraging, and data-driven AI career coach.\n"
            "You help candidates discover high-fit roles, close skill gaps, tailor applications, and refine job preferences.\n"
            "Always use your tools (`search_jobs`, `get_job_details`, `analyze_skill_gap`, `save_user_preference`) "
            "whenever the user asks about jobs, roles, skill gaps, or expresses preferences like remote work, salary, location, or target titles.\n"
            "Be concise, cite specific Job IDs and match percentages from tool outputs, and never invent job postings.\n\n"
            f"Candidate Profile:\n{json.dumps(profile_block, indent=2)}\n\n"
            f"Saved User Preferences:\n{json.dumps(preferences.model_dump(), indent=2)}"
        )

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]],
        profile: ResumeProfile | None,
        preferences: UserPreferences,
    ) -> AgentResponse:
        """Process a user message using OpenAI tool calling or deterministic fallback."""
        if not self.api_key:
            return self._run_fallback_agent(user_message, profile, preferences)

        try:
            client = OpenAI(api_key=self.api_key)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self._build_system_prompt(profile, preferences)}
            ]
            for msg in history[-10:]:
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            tool_calls_made: list[dict[str, Any]] = []
            collected_matches: list[JobMatch] = []
            any_prefs_updated = False

            for _ in range(4):
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
                choice = response.choices[0].message

                if not choice.tool_calls:
                    dedup_matches = self._dedup_matches(collected_matches)
                    return AgentResponse(
                        reply=choice.content or "Here is what I found for your career search.",
                        tool_calls_made=tool_calls_made,
                        referenced_jobs=dedup_matches[:5],
                        preferences_updated=any_prefs_updated,
                        mode="openai",
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.tool_calls
                        ],
                    }
                )

                for tc in choice.tool_calls:
                    try:
                        parsed_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        parsed_args = {}

                    result, changed = self._execute_tool(
                        name=tc.function.name,
                        args=parsed_args,
                        profile=profile,
                        preferences=preferences,
                        collected_matches=collected_matches,
                    )
                    any_prefs_updated = any_prefs_updated or changed
                    tool_calls_made.append(
                        {"tool": tc.function.name, "arguments": parsed_args}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )

            dedup_matches = self._dedup_matches(collected_matches)
            return AgentResponse(
                reply="I ran the requested job and skill analysis. Review the top matches below.",
                tool_calls_made=tool_calls_made,
                referenced_jobs=dedup_matches[:5],
                preferences_updated=any_prefs_updated,
                mode="openai",
            )
        except Exception as exc:
            logger.warning("OpenAI agent call failed (%s); using fallback career agent.", exc)
            return self._run_fallback_agent(user_message, profile, preferences)

    def _run_fallback_agent(
        self,
        user_message: str,
        profile: ResumeProfile | None,
        preferences: UserPreferences,
    ) -> AgentResponse:
        """Deterministic tool-calling fallback when OpenAI API key is unavailable."""
        lowered = user_message.lower()
        tool_calls_made: list[dict[str, Any]] = []
        collected_matches: list[JobMatch] = []
        prefs_updated = False

        # Detect preference updates from natural language
        pref_args: dict[str, Any] = {}
        if "remote" in lowered:
            pref_args["work_modes"] = ["Remote"]
        elif "hybrid" in lowered:
            pref_args["work_modes"] = ["Hybrid"]
        elif "onsite" in lowered or "on-site" in lowered:
            pref_args["work_modes"] = ["Onsite"]

        sal_match = re.search(r"\$?\s*(\d{2,3})\s*k\b|\$?\s*(\d{5,6})\b", lowered)
        if sal_match:
            val = int(sal_match.group(1)) * 1000 if sal_match.group(1) else int(sal_match.group(2))
            pref_args["min_salary"] = val

        for level in ("Entry", "Mid", "Senior", "Lead"):
            if re.search(rf"\b{level.lower()}\b", lowered):
                pref_args["experience_levels"] = [level]
                break

        if pref_args and any(
            kw in lowered for kw in ["prefer", "only", "want", "looking for", "target", "remember", "save", "minimum", "at least"]
        ):
            _, changed = self._execute_tool(
                "save_user_preference", pref_args, profile, preferences, collected_matches
            )
            prefs_updated = changed
            tool_calls_made.append({"tool": "save_user_preference", "arguments": pref_args})

        # Check specific job ID mention
        job_id_match = re.search(r"\b(JOB-\d{3})\b", user_message, re.I)
        if job_id_match:
            jid = job_id_match.group(1).upper()
            details, _ = self._execute_tool(
                "get_job_details", {"job_id": jid}, profile, preferences, collected_matches
            )
            tool_calls_made.append({"tool": "get_job_details", "arguments": {"job_id": jid}})
            if "error" not in details:
                matched = ", ".join(details["matched_skills"]) or "None detected yet"
                missing = ", ".join(details["missing_skills"]) or "None — strong match!"
                reply = (
                    f"### {details['job_id']}: {details['title']} at {details['company']}\n"
                    f"- **Location / Mode:** {details['location']} ({details['work_mode']}) · **{details['experience_level']}**\n"
                    f"- **Salary:** {details['salary_range']} · **Fit Score:** {details['overall_score']}%\n"
                    f"- **Matched Skills:** {matched}\n"
                    f"- **Missing Skills:** {missing}\n\n"
                    f"**Role Summary:** {details['description']}"
                )
                return AgentResponse(
                    reply=reply,
                    tool_calls_made=tool_calls_made,
                    referenced_jobs=self._dedup_matches(collected_matches)[:5],
                    preferences_updated=prefs_updated,
                    mode="fallback",
                )

        # Skill gap analysis intent
        if any(kw in lowered for kw in ["gap", "missing", "learn", "improve", "upskill", "weakness"]):
            gap_result, _ = self._execute_tool(
                "analyze_skill_gap",
                {"target": user_message},
                profile,
                preferences,
                collected_matches,
            )
            tool_calls_made.append({"tool": "analyze_skill_gap", "arguments": {"target": user_message}})
            gaps = gap_result["priority_missing_skills"]
            matched = gap_result["candidate_matched_skills"]
            reply = (
                "Based on your profile and top semantic matches, here is your skill gap analysis:\n\n"
                f"- **Strongest Matched Skills:** {', '.join(matched[:8]) or 'Upload a resume to unlock skill matching'}\n"
                f"- **Highest-Impact Missing Skills:** {', '.join(gaps[:6]) or 'No major gaps found'}\n\n"
                "**Recommended next step:** Build one focused portfolio project or add measurable impact bullets covering "
                f"**{', '.join(gaps[:3]) if gaps else 'system design & evaluation'}**."
            )
            return AgentResponse(
                reply=reply,
                tool_calls_made=tool_calls_made,
                referenced_jobs=self._dedup_matches(collected_matches)[:5],
                preferences_updated=prefs_updated,
                mode="fallback",
            )

        # Default: run semantic search_jobs tool
        search_args: dict[str, Any] = {
            "query": user_message,
            "work_mode": pref_args.get("work_modes", [None])[0] if pref_args.get("work_modes") else None,
            "experience_level": pref_args.get("experience_levels", [None])[0]
            if pref_args.get("experience_levels")
            else None,
            "min_salary": pref_args.get("min_salary"),
            "top_k": 5,
        }
        search_out, _ = self._execute_tool(
            "search_jobs", search_args, profile, preferences, collected_matches
        )
        tool_calls_made.append({"tool": "search_jobs", "arguments": search_args})

        lines = []
        if prefs_updated:
            lines.append("I updated your saved job preferences in memory.\n")

        results = search_out["results"]
        if not results:
            lines.append("No exact matches met all strict filters. Try broadening work mode or salary range.")
        else:
            lines.append(f"I searched the job index using `search_jobs()` and found **{len(results)}** top matches:\n")
            for item in results:
                missing_str = ", ".join(item["missing_skills"][:3]) or "None"
                lines.append(
                    f"- **{item['job_id']} · {item['title']}** at *{item['company']}* "
                    f"({item['work_mode']}, {item['salary_range']}) — **{item['overall_score']}% fit** "
                    f"| Missing: {missing_str}"
                )
            lines.append("\nAsk me about any `JOB-XXX` ID for a deep dive, or ask for a skill-gap roadmap!")

        return AgentResponse(
            reply="\n".join(lines),
            tool_calls_made=tool_calls_made,
            referenced_jobs=self._dedup_matches(collected_matches)[:5],
            preferences_updated=prefs_updated,
            mode="fallback",
        )

    @staticmethod
    def _dedup_matches(matches: list[JobMatch]) -> list[JobMatch]:
        seen: set[str] = set()
        unique: list[JobMatch] = []
        for m in matches:
            if m.job.job_id not in seen:
                seen.add(m.job.job_id)
                unique.append(m)
        return unique
