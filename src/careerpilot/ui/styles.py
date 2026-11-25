"""Custom CSS styling for the CareerPilot AI Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
/* Global layout polish */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1240px;
}

/* Hero header */
.cp-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #312E81 100%);
    color: #F8FAFC;
    padding: 1.4rem 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 10px 25px -10px rgba(15, 23, 42, 0.35);
}
.cp-hero h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.cp-hero p {
    margin: 0.35rem 0 0 0;
    color: #CBD5E1;
    font-size: 0.96rem;
}
.cp-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}
.cp-status-pill {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.12);
    color: #E2E8F0;
    border: 1px solid rgba(255, 255, 255, 0.18);
}
.cp-status-pill.live {
    background: rgba(16, 185, 129, 0.18);
    color: #6EE7B7;
    border-color: rgba(16, 185, 129, 0.4);
}

/* Metric cards */
.cp-kpi {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.cp-kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
}
.cp-kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 0.2rem;
}
.cp-kpi-sub {
    font-size: 0.8rem;
    color: #475569;
    margin-top: 0.15rem;
}

/* Job cards */
.cp-job-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.cp-job-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}
.cp-job-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}
.cp-job-meta {
    font-size: 0.86rem;
    color: #475569;
    margin-top: 0.2rem;
}
.cp-score-box {
    text-align: center;
    min-width: 82px;
    padding: 0.45rem 0.65rem;
    border-radius: 12px;
    font-weight: 700;
}
.cp-score-high {
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
}
.cp-score-mid {
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
}
.cp-score-low {
    background: #FFF7ED;
    color: #C2410C;
    border: 1px solid #FED7AA;
}

/* Skill pills */
.cp-pill {
    display: inline-block;
    padding: 0.18rem 0.58rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 999px;
    font-size: 0.77rem;
    font-weight: 600;
}
.cp-pill-match {
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}
.cp-pill-miss {
    background: #FEF2F2;
    color: #991B1B;
    border: 1px solid #FECACA;
}
.cp-pill-neutral {
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
}
.cp-pill-indigo {
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject dashboard styling into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
