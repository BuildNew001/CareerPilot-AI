"""Resume text extraction (PDF/plain text) and structured LLM/fallback parsing."""

from __future__ import annotations

import io
import logging
import re
from typing import BinaryIO, Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from careerpilot.models import ResumeProfile

logger = logging.getLogger(__name__)

# Curated skill taxonomy used for deterministic fallback extraction & normalization
CORE_SKILLS_TAXONOMY: list[str] = [
    "Python",
    "TypeScript",
    "JavaScript",
    "SQL",
    "Go",
    "Java",
    "Scala",
    "Rust",
    "C++",
    "Bash",
    "R",
    "OpenAI API",
    "Prompt Engineering",
    "Function Calling",
    "Tool Calling",
    "RAG",
    "LLM Architecture",
    "LLM Evaluation",
    "LLM Security",
    "Multimodal LLMs",
    "Embeddings",
    "Vector Databases",
    "Information Retrieval",
    "Search & Relevance",
    "NLP",
    "Machine Learning",
    "Deep Learning",
    "Transformers",
    "Diffusion Models",
    "Computer Vision",
    "Feature Engineering",
    "Statistics",
    "Causal Inference",
    "Experimentation",
    "A/B Testing",
    "Time Series",
    "Clustering",
    "Forecasting",
    "Optimization",
    "Bandits",
    "Attribution Modeling",
    "Survival Analysis",
    "Product Analytics",
    "Data Modeling",
    "Data Analysis",
    "Data Visualization",
    "Data Governance",
    "ETL",
    "System Design",
    "Distributed Systems",
    "REST APIs",
    "GraphQL",
    "gRPC",
    "Asyncio",
    "WebSockets",
    "OAuth2",
    "Microservices",
    "CI/CD",
    "Observability",
    "Application Security",
    "Threat Modeling",
    "OWASP",
    "API Security",
    "IAM",
    "Security Architecture",
    "HIPAA Compliance",
    "Product Strategy",
    "Product Management",
    "User Research",
    "Roadmapping",
    "Technical Writing",
    "Design Systems",
    "Performance Optimization",
    "Agile",
]

TOOLS_AND_FRAMEWORKS_TAXONOMY: list[str] = [
    "Streamlit",
    "FastAPI",
    "Django",
    "Flask",
    "React",
    "Next.js",
    "Node.js",
    "Tailwind CSS",
    "HTML/CSS",
    "Sentence Transformers",
    "Scikit-learn",
    "PyTorch",
    "TensorFlow",
    "Hugging Face",
    "LangGraph",
    "LangChain",
    "XGBoost",
    "Pandas",
    "NumPy",
    "Spacy",
    "OpenCV",
    "ONNX",
    "TensorRT",
    "CUDA",
    "Elasticsearch",
    "PostgreSQL",
    "Redis",
    "Snowflake",
    "BigQuery",
    "Databricks",
    "Delta Lake",
    "Apache Spark",
    "PySpark",
    "Kafka",
    "Flink",
    "Airflow",
    "dbt",
    "MLflow",
    "Docker",
    "Kubernetes",
    "Terraform",
    "AWS",
    "Azure",
    "GCP",
    "Linux",
    "Git",
    "GitHub Actions",
    "Pytest",
    "Playwright",
    "Jest",
    "Prometheus",
    "Grafana",
    "Datadog",
    "Looker",
    "Tableau",
    "Power BI",
    "Figma",
    "D3.js",
    "Plotly",
    "Ray",
    "Qdrant",
]

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Artificial Intelligence & LLMs": [
        "llm",
        "openai",
        "rag",
        "agent",
        "generative ai",
        "prompt",
        "langchain",
        "langgraph",
    ],
    "Semantic Search & Information Retrieval": [
        "semantic search",
        "embeddings",
        "vector",
        "retrieval",
        "elasticsearch",
        "ranking",
    ],
    "Data Science & Experimentation": [
        "data scien",
        "a/b test",
        "churn",
        "statistics",
        "causal",
        "xgboost",
        "forecasting",
    ],
    "Backend & Platform Engineering": [
        "fastapi",
        "django",
        "postgresql",
        "microservices",
        "rest api",
        "distributed",
        "redis",
    ],
    "Full-Stack & Product UI": [
        "react",
        "typescript",
        "next.js",
        "frontend",
        "full-stack",
        "streamlit",
        "ui/ux",
    ],
    "MLOps & Cloud Infrastructure": [
        "kubernetes",
        "docker",
        "terraform",
        "mlflow",
        "ci/cd",
        "aws",
        "sre",
    ],
}


class _OpenAIResumeSchema(BaseModel):
    """Structured output schema passed to OpenAI chat completions."""

    candidate_name: str = Field(description="Candidate full name or 'Candidate' if unknown")
    current_title: str = Field(description="Most recent or target job title")
    years_of_experience: float = Field(description="Total years of relevant experience")
    seniority_level: Literal["Entry", "Mid", "Senior", "Lead", "Executive"] = Field(
        description="Seniority tier inferred from experience"
    )
    skills: list[str] = Field(description="Core technical, analytical, and domain skills")
    tools_and_frameworks: list[str] = Field(
        description="Libraries, frameworks, databases, and cloud platforms"
    )
    domains: list[str] = Field(description="Industry or technical domains")
    education: list[str] = Field(description="Degrees, universities, or key certifications")
    career_interests: list[str] = Field(description="Target roles or career focus areas")
    summary: str = Field(description="Concise 2-3 sentence summary of candidate strengths")


def extract_text_from_pdf(file_input: bytes | BinaryIO) -> str:
    """Extract clean text from an uploaded PDF byte payload or file-like stream."""
    stream = io.BytesIO(file_input) if isinstance(file_input, (bytes, bytearray)) else file_input
    reader = PdfReader(stream)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(p.strip() for p in pages if p.strip())
    return text.strip()


def _match_taxonomy(text: str, taxonomy: list[str]) -> list[str]:
    """Find taxonomy entries present in text using boundary-aware case-insensitive matching."""
    lowered = text.lower()
    found: list[str] = []
    for skill in taxonomy:
        escaped = re.escape(skill.lower())
        # Allow punctuation boundaries like C++, CI/CD, Node.js
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(skill)
    return found


def _infer_years_of_experience(text: str) -> float:
    """Estimate years of experience from explicit mentions or date ranges."""
    explicit = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", text, re.I)
    if explicit:
        return max(float(v) for v in explicit)

    generic = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, re.I)
    if generic:
        vals = [float(v) for v in generic if float(v) <= 40.0]
        if vals:
            return max(vals)

    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if len(years) >= 2:
        span = max(years) - min(years)
        if 0 < span <= 40:
            return float(span)
    return 2.0


def _infer_seniority(years: float, text: str) -> Literal["Entry", "Mid", "Senior", "Lead", "Executive"]:
    lowered = text.lower()
    if re.search(r"\b(vp|vice president|cto|ceo|chief|head of)\b", lowered):
        return "Executive"
    if re.search(r"\b(staff|principal|lead|architect)\b", lowered):
        return "Lead"
    if years >= 5.0 or re.search(r"\b(senior|sr\.?)\b", lowered):
        return "Senior"
    if years < 2.0 or re.search(r"\b(junior|jr\.?|intern|entry)\b", lowered):
        return "Entry"
    return "Mid"


def extract_resume_fallback(resume_text: str) -> ResumeProfile:
    """Deterministic local resume analyzer used when no OpenAI API key is configured."""
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    candidate_name = "Candidate"
    if lines:
        first = lines[0]
        if len(first.split()) <= 5 and "@" not in first and ":" not in first:
            candidate_name = first

    current_title = "AI / Software Engineer"
    for line in lines[:8]:
        match = re.search(r"target role\s*:\s*(.+)", line, re.I)
        if match:
            current_title = match.group(1).strip()
            break
        if "|" in line and any(
            role in line.lower()
            for role in ["engineer", "scientist", "developer", "architect", "analyst", "manager"]
        ):
            current_title = line.split("|")[0].strip()
            break

    years = _infer_years_of_experience(resume_text)
    seniority = _infer_seniority(years, resume_text)
    skills = _match_taxonomy(resume_text, CORE_SKILLS_TAXONOMY)
    tools = _match_taxonomy(resume_text, TOOLS_AND_FRAMEWORKS_TAXONOMY)

    lowered = resume_text.lower()
    domains = [
        domain
        for domain, keywords in DOMAIN_KEYWORDS.items()
        if any(kw in lowered for kw in keywords)
    ]

    education = [
        line
        for line in lines
        if any(
            deg in line.lower()
            for deg in ["b.s.", "m.s.", "ph.d", "bachelor", "master", "university", "b.tech", "m.tech"]
        )
    ][:3]

    career_interests = [current_title] if current_title else []
    for domain in domains[:3]:
        if domain not in career_interests:
            career_interests.append(domain)

    top_skills = ", ".join((skills + tools)[:6]) or "software engineering"
    summary = (
        f"{seniority}-level professional ({years:g} yrs experience) focused on {current_title}. "
        f"Demonstrated strengths in {top_skills}."
    )

    return ResumeProfile(
        candidate_name=candidate_name,
        current_title=current_title,
        years_of_experience=years,
        seniority_level=seniority,
        skills=skills,
        tools_and_frameworks=tools,
        domains=domains,
        education=education,
        career_interests=career_interests,
        summary=summary,
        raw_text=resume_text,
        extraction_mode="fallback",
    )


def analyze_resume(
    resume_text: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> ResumeProfile:
    """Analyze a resume using OpenAI Structured Outputs with safe local fallback.

    Args:
        resume_text: Raw resume text extracted from PDF or pasted by user.
        api_key: Optional OpenAI API key. If absent, uses local taxonomy extractor.
        model: OpenAI model identifier supporting structured outputs.

    Returns:
        Validated ResumeProfile instance.
    """
    cleaned = resume_text.strip()
    if not cleaned:
        raise ValueError("Resume text cannot be empty.")

    if not api_key:
        return extract_resume_fallback(cleaned)

    try:
        client = OpenAI(api_key=api_key)
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical recruiter and ATS resume parser. "
                        "Extract clean, accurate structured candidate data from the resume. "
                        "Only extract skills and experience genuinely supported by the text."
                    ),
                },
                {"role": "user", "content": cleaned[:16000]},
            ],
            response_format=_OpenAIResumeSchema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned empty structured resume output.")

        return ResumeProfile(
            candidate_name=parsed.candidate_name,
            current_title=parsed.current_title,
            years_of_experience=max(0.0, float(parsed.years_of_experience)),
            seniority_level=parsed.seniority_level,
            skills=parsed.skills,
            tools_and_frameworks=parsed.tools_and_frameworks,
            domains=parsed.domains,
            education=parsed.education,
            career_interests=parsed.career_interests,
            summary=parsed.summary,
            raw_text=cleaned,
            extraction_mode="openai",
        )
    except Exception as exc:
        logger.warning("OpenAI resume parsing failed (%s); falling back to local extractor.", exc)
        return extract_resume_fallback(cleaned)
