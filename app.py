"""
AI Resume Analyzer & Job Recommendation System
Enterprise Talent Intelligence & Career Diagnostic Platform
"""

import os
# Set thread limits before importing numpy/sklearn on Windows to prevent OpenBLAS allocation errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from text_cleaner import clean_text, extract_sections
from resume_parser import parse_resume, validate_file
from skill_extractor import get_skill_extractor
from recommender import get_job_recommender
from ats_scorer import get_ats_scorer
from gap_analyzer import get_gap_analyzer
from resume_formatter import render_dyp_resume_html, DEFAULT_PROFILE
from resume_standardizer import get_resume_standardizer
import streamlit.components.v1 as components
import auth_manager


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ALZ ai | AI Resume Intelligence & Career Matcher",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Cached Backend Singletons
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing NLP & ML recommendation models...")
def init_models():
    skill_ext = get_skill_extractor()
    recommender = get_job_recommender()
    ats_eval = get_ats_scorer()
    gap_eval = get_gap_analyzer()
    standardizer = get_resume_standardizer()
    return skill_ext, recommender, ats_eval, gap_eval, standardizer


# -----------------------------------------------------------------------------
# Corporate Executive Black & Blue CSS Design System
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Import Executive Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #070b14;
    --bg-surface: #0f172a;
    --bg-card: #111c33;
    --border-subtle: #1e293b;
    --border-highlight: #2563eb;
    --border-hover: #3b82f6;
    --accent-blue: #2563eb;
    --accent-blue-light: #3b82f6;
    --accent-blue-dark: #1d4ed8;
    --accent-cyan: #38bdf8;
    --accent-tint: rgba(37, 99, 235, 0.14);
    --accent-amber: #f59e0b;
    --accent-rose: #f87171;
    --text-main: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
}

/* Global Reset */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f8fafc;
    background-color: #070b14;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
}

/* App Background: Midnight Obsidian Black & Sapphire Glow */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0d1b38 0%, #070b14 55%, #05080f 100%);
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Executive Header Card */
.executive-banner {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-top: 4px solid #2563eb;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.6), 0 0 20px -5px rgba(37, 99, 235, 0.2);
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.banner-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}

.banner-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.55;
}

/* Metric KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 22px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 6px 14px -2px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #2563eb, #38bdf8);
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6;
    box-shadow: 0 12px 24px -4px rgba(37, 99, 235, 0.3);
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #38bdf8;
    display: flex;
    align-items: baseline;
    gap: 6px;
    letter-spacing: -0.02em;
}

.kpi-subtext {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 6px;
}

/* Panels */
.pro-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 6px 16px -4px rgba(0, 0, 0, 0.5);
    transition: all 0.25s ease;
}

.pro-card:hover {
    border-color: #2563eb;
    box-shadow: 0 10px 24px -4px rgba(37, 99, 235, 0.2);
}

/* Role Item Card */
.role-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 4px solid #2563eb;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 6px 16px -4px rgba(0, 0, 0, 0.4);
}

.role-box:hover {
    border-color: #3b82f6;
    border-left: 4px solid #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px -4px rgba(37, 99, 235, 0.25);
}

.role-header-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.role-name {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
}

.score-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 9999px;
    letter-spacing: -0.02em;
}

/* Badges and Chips */
.tag-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.01em;
}

.chip-match {
    background: rgba(37, 99, 235, 0.16);
    border: 1px solid #2563eb;
    color: #60a5fa;
}

.chip-gap {
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #f87171;
}

.chip-tech {
    background: rgba(14, 165, 233, 0.14);
    border: 1px solid rgba(14, 165, 233, 0.35);
    color: #38bdf8;
}

.chip-neutral {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
}

/* Pillar Progress Item */
.pillar-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}

/* Milestone Phased Card */
.phase-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 4px solid #2563eb;
    box-shadow: 0 6px 16px -4px rgba(0, 0, 0, 0.4);
}

.phase-card.phase-2 { border-left-color: #3b82f6; }
.phase-card.phase-3 { border-left-color: #38bdf8; }

/* Streamlit Tabs Navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0f172a;
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    margin-bottom: 22px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 9px 18px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #94a3b8;
    border: none !important;
    background: transparent;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #60a5fa;
    background: rgba(37, 99, 235, 0.1);
}

.stTabs [aria-selected="true"] {
    background: rgba(37, 99, 235, 0.2) !important;
    color: #60a5fa !important;
    border: 1.5px solid #2563eb !important;
    font-weight: 700;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.25);
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 10px 22px;
    font-size: 0.92rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    transform: translateY(-1.5px);
    box-shadow: 0 8px 22px rgba(37, 99, 235, 0.45);
    border-color: rgba(255, 255, 255, 0.3);
}

/* Download Buttons */
div.stDownloadButton > button {
    background: #0f172a;
    color: #60a5fa;
    border: 1.5px solid #2563eb;
    font-weight: 600;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}

div.stDownloadButton > button:hover {
    background: rgba(37, 99, 235, 0.15);
    border-color: #3b82f6;
    color: #93c5fd;
    transform: translateY(-1px);
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #0b1120;
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] hr {
    margin: 16px 0;
    border-color: #1e293b;
}

/* Inputs & Selectboxes */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
    color: #f8fafc !important;
    background-color: #0b1120 !important;
}

.stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Chart Helper Functions (Midnight Black & Electric Blue Palette)
# -----------------------------------------------------------------------------
def build_ats_gauge_chart(score: float, grade: str, tier: str) -> go.Figure:
    """Builds an executive ATS gauge chart with midnight dark & electric blue theme."""
    if score >= 85:
        bar_color = "#3b82f6"
    elif score >= 70:
        bar_color = "#60a5fa"
    elif score >= 55:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={
                "text": f"<b>ATS Health Score: Grade {grade}</b><br><span style='font-size:12px;color:#94a3b8'>{tier}</span>",
                "font": {"size": 17, "color": "#f8fafc", "family": "Plus Jakarta Sans"},
            },
            number={"suffix": "/100", "font": {"size": 40, "color": "#f8fafc", "family": "Plus Jakarta Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155", "tickfont": {"color": "#94a3b8", "size": 10}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#0b1120",
                "borderwidth": 1,
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, 54], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [54, 69], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [69, 84], "color": "rgba(37, 99, 235, 0.16)"},
                    {"range": [84, 100], "color": "rgba(59, 130, 246, 0.24)"},
                ],
                "threshold": {
                    "line": {"color": "#38bdf8", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=25, r=25, t=35, b=10),
    )
    return fig


def build_skill_category_radar(skills_by_category: Dict[str, List[str]]) -> go.Figure:
    """Builds a radar chart reflecting candidate's skill breadth across categories in electric blue."""
    categories = list(skills_by_category.keys())
    counts = [len(skills_by_category[cat]) for cat in categories]

    if categories:
        categories_closed = categories + [categories[0]]
        counts_closed = counts + [counts[0]]
    else:
        categories_closed = []
        counts_closed = []

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=counts_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.22)",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6, color="#60a5fa"),
            name="Competencies",
            hovertemplate="<b>%{theta}</b>: %{r} Verified Skills<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(counts + [3]) + 1], tickfont=dict(color="#94a3b8", size=9), gridcolor="#1e293b"),
            angularaxis=dict(tickfont=dict(color="#f8fafc", size=10, family="Plus Jakarta Sans"), gridcolor="#1e293b"),
            bgcolor="#0f172a",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        showlegend=False,
        margin=dict(l=40, r=40, t=25, b=25),
    )
    return fig


def build_roles_comparison_chart(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """Builds a horizontal comparison bar chart for top matching job roles with electric blue palette."""
    top_roles = list(reversed(recommendations[:6]))
    role_names = [r["role_name"] for r in top_roles]
    match_scores = [r["match_score"] for r in top_roles]
    skill_scores = [r["skill_score"] for r in top_roles]
    semantic_scores = [r["semantic_score"] for r in top_roles]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=role_names,
            x=match_scores,
            orientation="h",
            name="Overall Match",
            marker=dict(color="#2563eb", line=dict(color="#3b82f6", width=1)),
            text=[f"{s}%" for s in match_scores],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Overall Match: %{x}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=role_names,
            x=skill_scores,
            orientation="h",
            name="Skill Overlap",
            marker=dict(color="#3b82f6", line=dict(color="#60a5fa", width=1)),
            hovertemplate="<b>%{y}</b><br>Skill Coverage: %{x}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=role_names,
            x=semantic_scores,
            orientation="h",
            name="Semantic Relevance",
            marker=dict(color="#38bdf8", line=dict(color="#93c5fd", width=1)),
            hovertemplate="<b>%{y}</b><br>Semantic TF-IDF: %{x}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#f8fafc", family="Plus Jakarta Sans", size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1", size=10)),
    )
    return fig


# -----------------------------------------------------------------------------
# Initialize Session State & User Authentication
# -----------------------------------------------------------------------------
if "resume_data" not in st.session_state:
    st.session_state["resume_data"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if "user_name" not in st.session_state:
    st.session_state["user_name"] = None


def render_auth_gateway():
    """Renders the executive White & Blue Login and Signup portal for ALZ ai."""
    st.markdown(
        """
        <style>
        /* White & Blue Theme for ALZ ai Login & Signup */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 35%, #e0f2fe 75%, #f0fdf4 100%) !important;
            color: #0f172a !important;
        }

        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f1f5f9 !important;
            border-radius: 12px !important;
            padding: 4px !important;
            border: 1px solid #e2e8f0 !important;
            gap: 6px !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: #64748b !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #1d4ed8 !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
            border: 1px solid #bfdbfe !important;
        }

        /* Text Input Labels & Fields */
        .stTextInput label {
            color: #1e3a8a !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            margin-bottom: 4px !important;
        }

        .stTextInput input {
            background-color: #ffffff !important;
            border: 1.5px solid #cbd5e1 !important;
            color: #0f172a !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            font-size: 0.93rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }

        .stTextInput input:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
            background-color: #ffffff !important;
        }

        .stTextInput input::placeholder {
            color: #94a3b8 !important;
        }

        /* Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 22px !important;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            transform: translateY(-1.5px) !important;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.45) !important;
        }
        </style>

        <div style="max-width:540px; margin: 24px auto 16px auto; text-align:center;">
            <div style="display:inline-flex; align-items:center; justify-content:center; background:linear-gradient(135deg, #2563eb 0%, #0284c7 100%); width:64px; height:64px; border-radius:18px; margin-bottom:12px; box-shadow:0 10px 25px rgba(37,99,235,0.32); border:2px solid #ffffff;">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
            </div>
            <div style="font-size:2.4rem; font-weight:900; color:#0f172a; letter-spacing:-0.03em; margin-bottom:2px; font-family:'Plus Jakarta Sans', sans-serif;">
                ALZ <span style="background:linear-gradient(135deg, #2563eb, #0284c7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">ai</span>
            </div>
            <div style="font-size:0.92rem; font-weight:700; color:#2563eb; letter-spacing:0.03em; text-transform:uppercase; margin-bottom:6px;">
                Next-Gen AI Resume & Placement Intelligence
            </div>
            <div style="font-size:0.88rem; color:#64748b; line-height:1.5;">
                Enterprise ATS Compliance Diagnostics & Dual-Factor Career Recommendation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_col_left, auth_col_center, auth_col_right = st.columns([1, 2.2, 1])
    with auth_col_center:
        auth_tab_login, auth_tab_signup = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with auth_tab_login:
            st.markdown(
                """
                <div style="font-size:0.9rem; color:#334155; margin-bottom:12px; margin-top:8px; font-weight:500;">
                    Sign in to your ALZ ai account to access your placement dashboard:
                </div>
                """,
                unsafe_allow_html=True,
            )
            login_email = st.text_input("Email Address", placeholder="name@company.com or candidate@gmail.com", key="auth_login_email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password", key="auth_login_password")

            btn_col1, btn_col2 = st.columns([1.3, 1])
            with btn_col1:
                if st.button("Sign In to ALZ ai", key="btn_do_login", use_container_width=True):
                    if not login_email or not login_password:
                        st.error("Please enter both email address and password.")
                    else:
                        success, message, user = auth_manager.authenticate_user(login_email, login_password)
                        if success and user:
                            st.session_state["authenticated"] = True
                            st.session_state["user_email"] = user["email"]
                            st.session_state["user_name"] = user["full_name"]
                            st.success("Authentication successful! Loading dashboard...")
                            time.sleep(0.4)
                            st.rerun()
                        else:
                            st.error(message)

            with btn_col2:
                if st.button("⚡ 1-Click Demo", key="btn_quick_demo", use_container_width=True):
                    success, message, user = auth_manager.authenticate_user("admin@alzai.com", "password123")
                    if not success:
                        success, message, user = auth_manager.authenticate_user("admin@talentiq.ai", "password123")
                    if success and user:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = user["email"]
                        st.session_state["user_name"] = user["full_name"]
                        st.rerun()

            st.markdown(
                """
                <div style="background:#eff6ff; border:1.5px solid #bfdbfe; border-radius:10px; padding:11px 16px; font-size:0.8rem; color:#1e40af; margin-top:18px; text-align:center;">
                    🔑 <b>1-Click Demo Account:</b> <code style="color:#1d4ed8; background:#dbeafe; padding:3px 8px; border-radius:6px; font-weight:700;">admin@alzai.com</code> | <code style="color:#1d4ed8; background:#dbeafe; padding:3px 8px; border-radius:6px; font-weight:700;">password123</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with auth_tab_signup:
            st.markdown(
                """
                <div style="font-size:0.9rem; color:#334155; margin-bottom:12px; margin-top:8px; font-weight:500;">
                    Create a new candidate or recruiter account on ALZ ai:
                </div>
                """,
                unsafe_allow_html=True,
            )
            signup_name = st.text_input("Full Name", placeholder="e.g. Ritesh Ture", key="auth_signup_name")
            signup_email = st.text_input("Email Address", placeholder="e.g. ritesh@example.com", key="auth_signup_email")
            signup_pwd = st.text_input("Create Password", type="password", placeholder="Minimum 6 characters", key="auth_signup_pwd")
            signup_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="auth_signup_confirm")

            if st.button("Create Account & Sign In", key="btn_do_signup", use_container_width=True):
                if not signup_email or not signup_pwd:
                    st.error("Email and password are required.")
                elif signup_pwd != signup_confirm:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    success, message = auth_manager.register_user(
                        email=signup_email,
                        password=signup_pwd,
                        full_name=signup_name,
                    )
                    if success:
                        _, _, user = auth_manager.authenticate_user(signup_email, signup_pwd)
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = signup_email.strip().lower()
                        st.session_state["user_name"] = signup_name.strip() or signup_email.split("@")[0].title()
                        st.success("Account created successfully! Loading dashboard...")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error(message)


# Check Authentication Gate
if not st.session_state.get("authenticated", False):
    render_auth_gateway()
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar: Controls & Document Ingestion (Authenticated Users Only)
# -----------------------------------------------------------------------------
skill_extractor, job_recommender, ats_scorer, gap_analyzer, resume_standardizer = init_models()

with st.sidebar:
    # Modern Blue Monogram for ALZ ai
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px; padding:4px 0;">
            <div style="background:linear-gradient(135deg, #2563eb, #0284c7); border:1px solid #38bdf8; width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 14px rgba(37,99,235,0.4);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
            </div>
            <div>
                <div style="font-family:'Plus Jakarta Sans'; font-weight:800; font-size:1.3rem; color:#f8fafc; letter-spacing:-0.02em;">ALZ <span style="color:#38bdf8;">ai</span></div>
                <div style="font-size:0.75rem; color:#93c5fd; font-weight:600;">Resume Intelligence & Matching</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active User Profile Badge & Sign Out Button
    cur_user_email = st.session_state.get("user_email", "admin@alzai.com")
    cur_user_name = st.session_state.get("user_name", "ALZ Administrator")
    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e3a5f; border-radius:10px; padding:10px 14px; margin-bottom:10px;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="overflow:hidden;">
                    <div style="font-size:0.82rem; font-weight:700; color:#f8fafc;">👤 {cur_user_name}</div>
                    <div style="font-size:0.72rem; color:#60a5fa; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{cur_user_email}</div>
                </div>
                <span style="background:rgba(37,99,235,0.2); border:1px solid #2563eb; color:#38bdf8; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:12px;">Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Sign Out", key="sidebar_sign_out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user_email"] = None
        st.session_state["user_name"] = None
        st.session_state["resume_data"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### Document Ingestion")

    # Sample resumes dropdown with clean corporate labels
    sample_files = {
        "-- Upload Candidate Resume --": None,
        "Ritesh Natharam Ture (AI & ML / Full Stack Profile)": "samples/ritesh_ture_resume.txt",
        "Senior AI & Machine Learning Engineer": "samples/ai_ml_engineer.txt",
        "Lead Full Stack Software Engineer": "samples/full_stack_developer.txt",
        "Senior Data Analyst & BI Specialist": "samples/data_analyst.txt",
    }
    selected_sample = st.selectbox("Load Sample Profile:", list(sample_files.keys()))

    if selected_sample != "-- Upload Candidate Resume --" and sample_files[selected_sample]:
        if st.button("Load Profile", use_container_width=True):
            sample_path = Path(sample_files[selected_sample])
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                file_name = sample_path.name
                parsed = parse_resume(file_name, file_bytes)
                st.session_state["resume_data"] = parsed
                st.session_state["file_name"] = file_name
                st.success(f"Profile {file_name} loaded.")
                st.rerun()

    st.markdown("<div style='text-align:center; font-size:0.78rem; color:#64748b; margin:6px 0;'>OR UPLOAD FILE</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="Upload candidate resume up to 10 MB in PDF, Word (DOCX), or plain text format.",
    )

    if uploaded_file is not None:
        if (
            st.session_state["file_name"] != uploaded_file.name
            or st.session_state["resume_data"] is None
        ):
            file_bytes = uploaded_file.read()
            is_valid, msg = validate_file(uploaded_file.name, len(file_bytes))
            if not is_valid:
                st.error(msg)
            else:
                try:
                    with st.spinner("Processing document & parsing structure..."):
                        parsed = parse_resume(uploaded_file.name, file_bytes)
                        st.session_state["resume_data"] = parsed
                        st.session_state["file_name"] = uploaded_file.name
                        st.success(f"Uploaded {uploaded_file.name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error parsing document: {str(e)}")

    st.markdown("---")
    st.markdown("#### Matching Configuration")

    top_n_roles = st.slider("Top Recommendations Count", min_value=1, max_value=10, value=5)
    skill_weight = st.slider(
        "Technical Skill Overlap Weight",
        min_value=0.2,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Controls the balance in hybrid scoring: Higher weight emphasizes hard skills coverage; lower weight balances semantic text relevance.",
    )

    min_match_filter = st.slider(
        "Minimum Match Score Filter (%)",
        min_value=30,
        max_value=80,
        value=40,
        step=5,
        help="Filter out positions below the designated threshold.",
    )

    if st.session_state["resume_data"] is not None:
        if st.button("Reset Document", use_container_width=True):
            st.session_state["resume_data"] = None
            st.session_state["file_name"] = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px; font-size:0.75rem; color:#94a3b8;">
            <b>Catalog Statistics:</b><br>
            • Benchmark Job Roles: <b style="color:#38bdf8;">{len(job_recommender.roles)}</b><br>
            • Technical Skills Indexed: <b style="color:#38bdf8;">{len(skill_extractor.skills_data)}</b><br>
            • Security: <b style="color:#60a5fa;">Authenticated Session</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
resume_data = st.session_state["resume_data"]

# EMPTY STATE / ONBOARDING SCREEN
if resume_data is None:
    st.markdown(
        """
        <div class="executive-banner">
            <div>
                <div class="banner-title">ALZ ai | Talent Intelligence & Resume Analytics Platform</div>
                <div class="banner-subtitle">
                    Automated ATS compliance scoring, hybrid semantic job matching, and phased competency roadmaps.
                    Benchmarked across industry talent acquisition models and technical hiring standards.
                </div>
            </div>
            <div>
                <span class="score-badge" style="background:rgba(37,99,235,0.2); color:#60a5fa; border:1px solid #2563eb;">
                    SYSTEM STATUS: READY
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="pro-card">
                <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:8px;">
                    Hybrid Role Recommendation
                </div>
                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                    Evaluates resumes against role descriptions using dual-factor scoring: exact technical skill set overlap and TF-IDF cosine semantic similarity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="pro-card">
                <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:8px;">
                    6-Pillar ATS Compliance Audit
                </div>
                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                    Systematic health diagnostics across section structure, action verb density, measurable impact metrics, formatting, and keyword saturation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="pro-card">
                <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:8px;">
                    Phased Upskilling Roadmap
                </div>
                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                    Maps candidate skill deltas against target benchmarks with structured 3-phase milestone timelines and hands-on portfolio deliverables.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns([1.2, 2])
    with c_btn1:
        if st.button("⚡ Launch Demo Analysis (Ritesh Profile)", use_container_width=True):
            sample_path = Path("samples/ritesh_ture_resume.txt")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                parsed = parse_resume("ritesh_ture_resume.txt", file_bytes)
                st.session_state["resume_data"] = parsed
                st.session_state["file_name"] = "ritesh_ture_resume.txt"
                st.rerun()
    with c_btn2:
        st.markdown(
            """
            <div style="font-size:0.86rem; color:#94a3b8; display:flex; align-items:center; height:100%; padding-left:8px;">
                👈 <span>Upload your <b>PDF, DOCX, or TXT</b> resume in the sidebar or select from the preloaded profiles.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# -----------------------------------------------------------------------------
# RUN FULL ANALYSIS PIPELINE: STANDARDIZE FIRST -> THEN EVALUATE & COMPARE
# -----------------------------------------------------------------------------
try:
    # STEP 1: CONVERT UPLOADED RESUME INTO THE STANDARDIZED PLACEMENT FORMAT
    std_result = resume_standardizer.standardize(
        raw_text=resume_data["raw_text"],
        file_name=resume_data["file_name"],
        sections=resume_data["sections"],
    )
    standardized_profile = std_result["profile"]
    standardized_text = std_result["formatted_text"]
    standardized_html = std_result["html"]
    candidate_name = std_result["name"]

    st.session_state["custom_resume_profile"] = standardized_profile
    st.session_state["standardized_html"] = standardized_html

    # Sections and Word Count for downstream diagnostics
    std_sections = extract_sections(standardized_text)
    sections = std_sections
    std_word_count = len(standardized_text.split())
    word_count = std_word_count

    # STEP 2: EXTRACT RELEVANT TECHNICAL SKILLS FROM THE STANDARDIZED RESUME
    skills_res = skill_extractor.extract_skills(standardized_text)
    matched_skills = skills_res.get("matched_skills", [])
    skills_by_category = skills_res.get("categorized_skills") or skills_res.get("by_category", {})

    # STEP 3: ATS COMPLIANCE AUDIT ON THE STANDARDIZED RESUME
    cleaned_std_text = clean_text(standardized_text)

    ats_res = ats_scorer.evaluate_resume(
        raw_text=standardized_text,
        cleaned_text=cleaned_std_text,
        sections=std_sections,
        extracted_skills=matched_skills,
        word_count=std_word_count,
    )

    # STEP 4: HYBRID JOB ROLE MATCHING & COMPARISON AGAINST BENCHMARK ROLES
    recommendations = job_recommender.match_roles(
        candidate_skills=matched_skills,
        resume_text=cleaned_std_text,
        top_n=top_n_roles,
        skill_weight=skill_weight,
        semantic_weight=round(1.0 - skill_weight, 2),
    )
except Exception as e:
    import traceback
    st.error(f"Error analyzing document: {str(e)}")
    with st.expander("Technical Diagnostic Details"):
        st.code(traceback.format_exc(), language="text")
    st.info("Please verify the document contains readable text and standard formatting.")
    st.stop()

top_role = recommendations[0] if recommendations else None

# Primary domain category
if skills_by_category:
    top_cat = max(skills_by_category.items(), key=lambda x: len(x[1]))[0]
else:
    top_cat = "General Engineering"


# -----------------------------------------------------------------------------
# EXECUTIVE TOP BANNER
# -----------------------------------------------------------------------------
candidate_name = resume_data["file_name"].replace(".txt", "").replace(".pdf", "").replace(".docx", "").replace("_", " ").title()

st.markdown(
    f"""
    <div class="executive-banner">
        <div>
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; color:#38bdf8; font-weight:700;">
                EXECUTIVE CANDIDATE AUDIT
            </div>
            <div class="banner-title" style="margin-top:2px;">
                {candidate_name}
            </div>
            <div class="banner-subtitle">
                Format: <b style="color:#f8fafc;">{resume_data['format']}</b> &nbsp;|&nbsp;
                Word Count: <b style="color:#f8fafc;">{word_count}</b> (~{round(word_count/250, 1)} min read) &nbsp;|&nbsp;
                Core Competency Domain: <b style="color:#38bdf8;">{top_cat}</b>
            </div>
        </div>
        <div>
            <span class="score-badge" style="background:rgba(37,99,235,0.2); color:#60a5fa; border:1.5px solid #2563eb; font-size:0.95rem; padding:8px 18px; box-shadow:0 2px 12px rgba(37,99,235,0.25);">
                ATS Rating: Grade {ats_res['grade']} ({ats_res['tier']})
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4 Executive KPI Cards
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">ATS Compliance Score</div>
            <div class="kpi-value" style="color:{ats_res['color']};">
                {ats_res['overall_score']}<span style="font-size:1rem; color:#94a3b8;">/100</span>
            </div>
            <div class="kpi-subtext">
                Grade <b>{ats_res['grade']}</b> • {ats_res['tier']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    top_role_title = top_role["role_name"] if top_role else "N/A"
    top_role_match = top_role["match_score"] if top_role else 0
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Primary Career Alignment</div>
            <div class="kpi-value" style="color:#38bdf8;">
                {top_role_match}%
            </div>
            <div class="kpi-subtext" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {top_role_title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Technical Skills Identified</div>
            <div class="kpi-value" style="color:#60a5fa;">
                {len(matched_skills)}<span style="font-size:1rem; color:#94a3b8;"> skills</span>
            </div>
            <div class="kpi-subtext">
                Across <b>{len(skills_by_category)}</b> domain specializations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    av_details = ats_res["breakdown"].get("action_verbs", {}).get("details", {})
    action_verbs_cnt = av_details.get("action_verbs_count", av_details.get("action_verb_count", 0))
    qi_details = ats_res["breakdown"].get("quantifiable_impact", {}).get("details", {})
    metrics_count = qi_details.get("metrics_count", 0)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Impact Verbs & Metrics</div>
            <div class="kpi-value" style="color:#38bdf8;">
                {action_verbs_cnt}<span style="font-size:1rem; color:#94a3b8;"> verbs</span>
            </div>
            <div class="kpi-subtext">
                <b>{metrics_count}</b> measurable impact metrics detected
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 2-STAGE ANALYSIS PIPELINE BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="background:linear-gradient(135deg, #0e1e3e 0%, #0f172a 100%); border:1.5px solid #1e3a5f; border-radius:14px; padding:16px 20px; margin: 18px 0 24px 0; box-shadow:0 4px 14px -2px rgba(37, 99, 235, 0.2);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:#2563eb; color:#ffffff; width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; font-weight:700;">
                    ✓
                </div>
                <div>
                    <div style="font-size:0.95rem; font-weight:700; color:#93c5fd; letter-spacing:-0.01em;">
                        Step 1 Complete: Resume Converted into Official Placement Format
                    </div>
                    <div style="font-size:0.8rem; color:#cbd5e1;">
                        Raw document normalized into D.Y. Patil placement standard before technical skill extraction and role matching.
                    </div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(37,99,235,0.25); border:1px solid #3b82f6; color:#93c5fd; padding:5px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">
                    <span style="width:6px; height:6px; border-radius:50%; background:#38bdf8;"></span>
                    Step 2 Active: Job Alignment & ATS Diagnostics
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# APPLICATION TABS (Professional Corporate Labels)
# -----------------------------------------------------------------------------
tab_formatter, tab_roles, tab_ats, tab_gap, tab_skills, tab_bullet, tab_export = st.tabs(
    [
        " Converted Placement Resume",
        " Job Matching & Role Alignment",
        " ATS Health Audit",
        " Skill Gap Roadmap",
        " Competency Matrix",
        " STAR Bullet Studio",
        " Executive Report",
    ]
)


# =============================================================================
# TAB 1: CONVERTED PLACEMENT RESUME (D.Y. PATIL FORMAT)
# =============================================================================
with tab_formatter:
    st.markdown(" Standardized Placement Resume Document ")
    st.markdown(
        "Candidate resume was first parsed, structured, and converted into the canonical **D.Y. Patil Placement Format** "
        "(cyan brand header, key skills badges, 2-column detail grid, and structured sections). "
        "Job descriptions and technical skill matrices are evaluated directly against this standardized format."
    )

    # Format Action Toolbar
    act_col1, act_col2, act_col3 = st.columns([1.5, 1.2, 1.3])
    with act_col1:
        st.download_button(
            label="📥 Download Standardized Resume (HTML)",
            data=standardized_html,
            file_name=f"{candidate_name.lower().replace(' ', '_')}_placement_resume.html",
            mime="text/html",
            use_container_width=True,
        )
    with act_col2:
        st.download_button(
            label="📄 Download Plaintext (.txt)",
            data=standardized_text,
            file_name=f"{candidate_name.lower().replace(' ', '_')}_standardized.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with act_col3:
        st.markdown(
            """
            <div style="font-size:0.82rem; color:#475569; display:flex; align-items:center; height:100%; padding-left:8px;">
                💡 <span>Print / PDF Tip: Open HTML in browser and press <b>Ctrl + P</b> &rarr; <i>Save as PDF</i></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Live Rendered Standardized Resume
    st.markdown(" Live Placement Document Preview ")
    components.html(standardized_html, height=880, scrolling=True)

    # Interactive Profile Customizer
    with st.expander("🛠️ Interactive Placement Profile Customizer (Edit & Re-Evaluate)"):
        st.markdown(
            "Tweak candidate details below to update the standardized placement resume and re-evaluate matches."
        )
        c_prof = st.session_state.get("custom_resume_profile", standardized_profile)

        edit_col1, edit_col2 = st.columns(2)
        with edit_col1:
            new_name = st.text_input("Candidate Name", value=c_prof.get("name", candidate_name), key="edit_cand_name")
            new_email = st.text_input("Email Address", value=c_prof.get("email", "candidate@gmail.com"), key="edit_cand_email")
            new_phone = st.text_input("Phone Number", value=c_prof.get("phone", "+91-9876543210"), key="edit_cand_phone")
        with edit_col2:
            new_role = st.text_input("Role / Degree Subtitle", value=c_prof.get("role_subtitle", "B.E. - Computer Science & Engineering"), key="edit_cand_role")
            new_loc = st.text_input("Current Address / Location", value=c_prof.get("location", "Pune, Maharashtra, India"), key="edit_cand_loc")

        if st.button("Apply Profile Changes & Re-Render Resume", key="apply_profile_edit", use_container_width=True):
            c_prof["name"] = new_name.upper()
            c_prof["email"] = new_email
            c_prof["phone"] = new_phone
            c_prof["role_subtitle"] = new_role
            c_prof["location"] = new_loc
            st.session_state["custom_resume_profile"] = c_prof
            st.session_state["standardized_html"] = render_dyp_resume_html(c_prof, enable_toolbar=True)
            st.success("Placement profile updated! Re-rendering...")
            st.rerun()

    # View Canonical Standardized Plaintext
    with st.expander("📋 View Canonical Standardized Plaintext (Input to ATS & Job Matcher)"):
        st.markdown("This clean, structured text representation is what powers our downstream ATS health audit and job matching models:")
        st.code(standardized_text, language="text")


# =============================================================================
# TAB 2: JOB MATCHING & ROLE ALIGNMENT
# =============================================================================
with tab_roles:
    st.markdown(" Career Role Alignment & Benchmark Ranking")
    st.markdown(
        "Standardized candidate profile evaluated against industry role specifications using hybrid ranking "
        f"({int(skill_weight*100)}% hard technical skill coverage, {int((1-skill_weight)*100)}% semantic TF-IDF relevance)."
    )

    # Custom Job Description Comparison Expander
    with st.expander("🔍 Compare Candidate with a Custom Job Description"):
        st.markdown(
            "Paste any specific job description from a company job posting or campus drive to compare against the standardized candidate resume."
        )
        custom_jd_title = st.text_input("Target Job Title / Designation:", placeholder="e.g. Full Stack Developer, Data Engineer", key="custom_jd_title")
        custom_jd_text = st.text_area("Paste Job Description / Requirements:", height=140, placeholder="Paste job requirements, required qualifications, and technical stack...", key="custom_jd_text")

        if st.button("Evaluate Match Against Custom Job Description", key="btn_eval_custom_jd"):
            if not custom_jd_text.strip():
                st.warning("Please paste a job description first.")
            else:
                jd_eval = job_recommender.evaluate_custom_jd(
                    job_title=custom_jd_title,
                    job_description=custom_jd_text,
                    candidate_skills=matched_skills,
                    resume_text=cleaned_std_text,
                    skill_weight=skill_weight,
                    semantic_weight=round(1.0 - skill_weight, 2),
                )
                st.markdown(f"#### Match Diagnostics for: **{jd_eval['job_title']}**")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Overall Fit Score", f"{jd_eval['match_score']}%", jd_eval["match_level"])
                res_col2.metric("Skill Coverage", f"{jd_eval['skill_score']}%")
                res_col3.metric("Semantic Relevance", f"{jd_eval['semantic_score']}%")

                st.markdown(f"**Verified Technical Skills ({len(jd_eval['matched_skills'])}):**")
                if jd_eval["matched_skills"]:
                    st.markdown(" ".join([f"<span class='tag-chip chip-match'>{s}</span>" for s in jd_eval["matched_skills"]]), unsafe_allow_html=True)
                else:
                    st.info("No direct skill matches found in this JD.")

                st.markdown(f"**Skill Gaps / Missing Competencies ({len(jd_eval['missing_skills'])}):**")
                if jd_eval["missing_skills"]:
                    st.markdown(" ".join([f"<span class='tag-chip chip-gap'>{s}</span>" for s in jd_eval["missing_skills"]]), unsafe_allow_html=True)
                else:
                    st.success("Candidate matches 100% of required technical skills in this job description!")

    st.markdown("---")

    # Interactive Controls Row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1])

    with ctrl_col1:
        categories = ["All Specializations"] + sorted(list(set(r["category"] for r in job_recommender.roles)))
        selected_cat = st.selectbox("Filter by Specialization:", categories)

    with ctrl_col2:
        search_role_keyword = st.text_input("Search Job Title:", placeholder="e.g. Engineer, Analyst")

    with ctrl_col3:
        display_mode = st.radio("Display Mode:", ["Role Cards", "Comparison Matrix"], horizontal=True)

    # Apply filters
    filtered_roles = [r for r in recommendations if r["match_score"] >= min_match_filter]
    if selected_cat != "All Specializations":
        filtered_roles = [r for r in filtered_roles if r["category"] == selected_cat]
    if search_role_keyword.strip():
        filtered_roles = [r for r in filtered_roles if search_role_keyword.lower() in r["role_name"].lower()]

    if not filtered_roles:
        st.warning(f"No roles matched the current filters (Category: '{selected_cat}', Min Match: {min_match_filter}%). Lower the match threshold in the sidebar to view more opportunities.")
        filtered_roles = recommendations[:3]

    col_chart, col_quick_kpi = st.columns([2, 1])
    with col_chart:
        fig_roles = build_roles_comparison_chart(filtered_roles)
        st.plotly_chart(fig_roles, use_container_width=True)

    with col_quick_kpi:
        st.markdown(
            f"""
            <div class="pro-card" style="padding:18px;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#94a3b8; font-weight:700;">Matching Overview</div>
                <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:6px 0;">
                    {len(filtered_roles)} Roles Meeting Criteria
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.5;">
                    Top match is <b style="color:#38bdf8;">{filtered_roles[0]['role_name']}</b> with <b style="color:#38bdf8;">{filtered_roles[0]['match_score']}%</b> overall alignment.
                </div>
                <div style="margin-top:12px; font-size:0.75rem; color:#94a3b8;">
                    • Threshold: <b>{min_match_filter}%</b><br>
                    • Weight: <b>{int(skill_weight*100)}% Skill / {int((1-skill_weight)*100)}% Semantic</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display Mode: Role Cards vs Comparison Matrix
    if display_mode == "Comparison Matrix":
        st.markdown("#### Comparative Benchmark Matrix")
        table_data = []
        for r in filtered_roles:
            table_data.append({
                "Role Title": r["role_name"],
                "Specialization": r["category"],
                "Overall Match": f"{r['match_score']}%",
                "Skill Coverage": f"{r['skill_score']}%",
                "Semantic Relevance": f"{r['semantic_score']}%",
                "Matched Count": len(r["matched_skills"]),
                "Missing Count": len(r["missing_skills"]),
            })
        df_matrix = pd.DataFrame(table_data)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
    else:
        st.markdown("#### Role Match Breakdown")
        for i, role in enumerate(filtered_roles, 1):
            m_score = role["match_score"]
            score_color = "#3b82f6" if m_score >= 60 else ("#60a5fa" if m_score >= 50 else "#f59e0b")

            matched_chips = " ".join([f"<span class='tag-chip chip-match'>{s}</span>" for s in role["matched_skills"]])
            missing_chips = " ".join([f"<span class='tag-chip chip-gap'>{s}</span>" for s in role["missing_skills"]]) if role["missing_skills"] else "<span style='color:#38bdf8; font-size:0.82rem; font-weight:600;'>100% Required Skill Alignment</span>"

            st.markdown(
                f"""
                <div class="role-box">
                    <div class="role-header-line">
                        <div>
                            <span style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase;">#{i} • {role['category']}</span>
                            <div class="role-name">{role['role_name']}</div>
                        </div>
                        <div class="score-badge" style="border:1.5px solid {score_color}66; background:rgba(37,99,235,0.18); color:{score_color}; font-weight:800;">
                            {m_score}% Match
                        </div>
                    </div>
                    <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:12px; display:flex; gap:16px; flex-wrap:wrap;">
                        <span>Skill Overlap: <b style="color:#f8fafc;">{role['skill_score']}%</b></span>
                        <span>Semantic Score: <b style="color:#f8fafc;">{role['semantic_score']}%</b></span>
                        <span>Matched: <b style="color:#38bdf8;">{len(role['matched_skills'])}</b> / {len(role['required_skills'])}</span>
                    </div>
                    <div style="margin-bottom:8px;">
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#38bdf8; font-weight:700; margin-bottom:4px;">Verified Competencies ({len(role['matched_skills'])})</div>
                        <div>{matched_chips}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem; text-transform:uppercase; color:#f87171; font-weight:700; margin-bottom:4px;">Recommended Competencies ({len(role['missing_skills'])})</div>
                        <div>{missing_chips}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Specification Details: {role['role_name']}"):
                st.write(role["description"])
                st.markdown(f"**Required Skill Set:** {', '.join(role['required_skills'])}")


# =============================================================================
# TAB 2: ATS HEALTH AUDIT & WHAT-IF SIMULATOR
# =============================================================================
with tab_ats:
    st.markdown("### Applicant Tracking System (ATS) Compliance Diagnostic")
    st.markdown("Algorithmic screening audit against enterprise parsing criteria, readability, and recruiter scanning parameters.")

    col_gauge, col_breakdown = st.columns([1, 1.3])

    with col_gauge:
        fig_ats = build_ats_gauge_chart(ats_res["overall_score"], ats_res["grade"], ats_res["tier"])
        st.plotly_chart(fig_ats, use_container_width=True)

        st.markdown(
            f"""
            <div class="pro-card" style="margin-top:10px;">
                <div style="font-weight:700; font-size:0.92rem; margin-bottom:4px; color:#f8fafc;">
                    Compliance Classification: <span style="color:{ats_res['color']}">{ats_res['tier']}</span>
                </div>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.5;">
                    Resumes scoring above 80 points typically bypass automated screening filters with a 92%+ pass rate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_breakdown:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; text-transform:uppercase; color:#94a3b8; margin-bottom:10px;'>ATS Health Pillars</div>", unsafe_allow_html=True)
        breakdown = ats_res["breakdown"]

        av_pillar = breakdown.get("action_verbs", {})
        av_det = av_pillar.get("details", {})
        av_cnt = av_det.get("action_verbs_count", av_det.get("action_verb_count", 0))

        qi_pillar = breakdown.get("quantifiable_impact", {})
        qi_det = qi_pillar.get("details", {})
        qi_cnt = qi_det.get("metrics_count", 0)

        lh_pillar = breakdown.get("length_health", breakdown.get("length_hygiene", {}))
        lh_det = lh_pillar.get("details", {})
        lh_stat = lh_det.get("status", lh_det.get("length_status", "Standard"))

        sec_pillar = breakdown.get("section_structure", {})
        ci_pillar = breakdown.get("contact_info", {})
        sd_pillar = breakdown.get("skill_density", {})

        pillars = [
            ("Section Architecture", sec_pillar.get("score", 0), sec_pillar.get("max", 25), f"{len(sections)} standard sections identified"),
            ("Action Verb Saturation", av_pillar.get("score", 0), av_pillar.get("max", 20), f"{av_cnt} dynamic verbs detected"),
            ("Quantified Evidence", qi_pillar.get("score", 0), qi_pillar.get("max", 20), f"{qi_cnt} metrics/percentages identified"),
            ("Document Volume", lh_pillar.get("score", 0), lh_pillar.get("max", 15), f"{word_count} words ({lh_stat})"),
            ("Contact & Portfolio", ci_pillar.get("score", 0), ci_pillar.get("max", 10), "Contact channels and links parsed"),
            ("Technical Keyword Saturation", sd_pillar.get("score", 0), sd_pillar.get("max", 10), f"{sd_pillar.get('details', {}).get('density_pct', 0)}% technical keyword density"),
        ]

        for p_name, p_score, p_max, p_detail in pillars:
            pct = min(100, int((p_score / p_max) * 100)) if p_max > 0 else 0
            bar_color = "#3b82f6" if pct >= 75 else ("#60a5fa" if pct >= 50 else "#f59e0b")
            st.markdown(
                f"""
                <div class="pillar-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:600; font-size:0.85rem; color:#f8fafc;">{p_name}</span>
                        <span style="font-family:'JetBrains Mono'; font-weight:700; font-size:0.88rem; color:{bar_color};">{p_score}/{p_max} pts</span>
                    </div>
                    <div style="background:#1e293b; border-radius:9999px; height:6px; overflow:hidden; margin-bottom:4px;">
                        <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:9999px;"></div>
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8;">{p_detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # INTERACTIVE WHAT-IF SIMULATOR
    st.markdown("---")
    st.markdown("#### Interactive What-If Optimization Simulator")
    st.markdown("Simulate proposed resume adjustments to project immediate score improvements before submitting your application.")

    sim_col1, sim_col2 = st.columns([1.2, 1])

    with sim_col1:
        st.markdown("**Simulated Improvements:**")
        sim_metrics = st.checkbox("Add 3+ Quantified Business Metrics / Percentages (+10 pts)", value=False)
        sim_verbs = st.checkbox("Replace Passive Language with High-Impact Action Verbs (+8 pts)", value=False)
        sim_links = st.checkbox("Ensure Active LinkedIn & GitHub Portfolio Links (+5 pts)", value=False)
        sim_sections = st.checkbox("Standardize Section Headings (Experience, Skills, Education) (+6 pts)", value=False)
        sim_keywords = st.checkbox("Inject 5+ Relevant Domain Keywords into Summary (+6 pts)", value=False)

    # Calculate simulated score
    sim_boost = 0
    if sim_metrics: sim_boost += 10
    if sim_verbs: sim_boost += 8
    if sim_links: sim_boost += 5
    if sim_sections: sim_boost += 6
    if sim_keywords: sim_boost += 6

    projected_score = min(100, ats_res["overall_score"] + sim_boost)
    projected_grade = "A" if projected_score >= 85 else ("B" if projected_score >= 70 else ("C" if projected_score >= 55 else "D"))

    with sim_col2:
        st.markdown(
            f"""
            <div class="pro-card" style="text-align:center; padding:20px;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#94a3b8; font-weight:700;">Simulated Projected Score</div>
                <div style="font-size:2.4rem; font-weight:800; color:#38bdf8; margin:8px 0; font-family:'Plus Jakarta Sans';">
                    {projected_score}<span style="font-size:1.1rem; color:#94a3b8;">/100</span>
                </div>
                <div style="font-size:0.85rem; color:#60a5fa; font-weight:700;">
                    +{sim_boost} Points Improvement (Grade {projected_grade})
                </div>
                <div style="font-size:0.78rem; color:#94a3b8; margin-top:8px;">
                    Base Score: {ats_res['overall_score']} → Projected Score: {projected_score}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action Verbs vs Quantifiable Metrics Details
    st.markdown("---")
    st.markdown("#### Impact Linguistics Diagnostic")

    col_verbs, col_weak = st.columns(2)

    with col_verbs:
        found_verbs = breakdown["action_verbs"]["details"].get("found_verbs", breakdown["action_verbs"]["details"].get("action_verbs_sample", []))
        st.markdown(f"**Identified Dynamic Action Verbs ({len(found_verbs)})**")
        if found_verbs:
            v_chips = " ".join([f"<span class='tag-chip chip-match'>{v}</span>" for v in sorted(found_verbs)])
            st.markdown(f"<div style='margin-bottom:12px;'>{v_chips}</div>", unsafe_allow_html=True)
        else:
            st.warning("No high-impact action verbs identified. Open bullet points with verbs such as 'architected', 'spearheaded', or 'automated'.")

    with col_weak:
        weak_details = breakdown["quantifiable_impact"]["details"].get("metrics_found", [])
        st.markdown(f"**Extracted Quantifiable Measurements ({len(weak_details)})**")
        if weak_details:
            m_chips = " ".join([f"<span class='tag-chip chip-tech'>{m}</span>" for m in weak_details[:10]])
            st.markdown(f"<div style='margin-bottom:12px;'>{m_chips}</div>", unsafe_allow_html=True)
        else:
            st.info("No quantifiable metrics detected. Add scale metrics like percentage increases, latency reductions, or team size.")

    # Optimization Checklist
    st.markdown("#### Priority Action Items")
    for idx, tip in enumerate(ats_res["suggestions"], 1):
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-start; gap:12px; background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:10px 14px; margin-bottom:8px; box-shadow:0 2px 8px rgba(0,0,0,0.3);">
                <span style="background:rgba(37,99,235,0.2); color:#60a5fa; border:1px solid #2563eb; border-radius:4px; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; flex-shrink:0;">{idx}</span>
                <span style="font-size:0.85rem; color:#cbd5e1; line-height:1.4;">{tip}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 3: SKILL GAP ANALYSIS & ROADMAP
# =============================================================================
with tab_gap:
    st.markdown("### Targeted Skill Gap Audit & Milestone Roadmap")
    st.markdown("Select any target role from the catalog to analyze competency gaps and simulate your upskilling progression.")

    # Target Role Selection
    all_role_names = [r["role_name"] for r in job_recommender.roles]
    default_role_idx = 0
    if top_role and top_role["role_name"] in all_role_names:
        default_role_idx = all_role_names.index(top_role["role_name"])

    selected_target_role_name = st.selectbox(
        "Select Target Benchmark Role:",
        all_role_names,
        index=default_role_idx,
    )

    target_role_obj = next(r for r in job_recommender.roles if r["role_name"] == selected_target_role_name)
    gap_result = gap_analyzer.analyze_gaps(target_role_obj, matched_skills)

    # Readiness Summary Banner
    st.markdown(
        f"""
        <div class="pro-card" style="border-left:4px solid #2563eb; margin-top:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:700;">Role Readiness Evaluation</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#f8fafc; margin-top:2px;">
                        {gap_result['role_name']} • <span style="color:#38bdf8;">{gap_result['coverage_pct']}% Qualified</span>
                    </div>
                </div>
                <div style="display:flex; gap:20px;">
                    <div style="text-align:center;">
                        <div style="font-size:1.3rem; font-weight:800; color:#38bdf8;">{gap_result['matched_count']}</div>
                        <div style="font-size:0.72rem; color:#94a3b8;">Matched Skills</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.3rem; font-weight:800; color:#ef4444;">{gap_result['missing_count']}</div>
                        <div style="font-size:0.72rem; color:#94a3b8;">Missing Skills</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.3rem; font-weight:800; color:#60a5fa;">~{gap_result['estimated_weeks_total']} wks</div>
                        <div style="font-size:0.72rem; color:#94a3b8;">Est. Upskill Time</div>
                    </div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:0.85rem; color:#cbd5e1; border-top:1px solid #1e293b; padding-top:10px;">
                {gap_result['readiness_summary']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # INTERACTIVE BRIDGE-THE-GAP SIMULATOR
    if gap_result["missing_skills"]:
        st.markdown("#### Interactive Bridge-the-Gap Simulator")
        st.markdown("Check off the skills you plan to acquire to view your projected role qualification coverage:")

        selected_to_learn = st.multiselect(
            "Select skills to add to your plan:",
            gap_result["missing_skills"],
            default=gap_result["missing_skills"][:1] if gap_result["missing_skills"] else [],
        )

        projected_coverage = round(
            ((len(gap_result["matched_skills"]) + len(selected_to_learn)) / gap_result["total_required"]) * 100, 1
        ) if gap_result["total_required"] > 0 else 100.0

        st.progress(projected_coverage / 100.0)
        st.markdown(
            f"**Projected Qualification:** <b style='color:#38bdf8;'>{projected_coverage}%</b> "
            f"(+{round(projected_coverage - gap_result['coverage_pct'], 1)}% increase)",
            unsafe_allow_html=True,
        )

    # Missing Skills Priority Breakdown
    if gap_result["missing_skill_details"]:
        st.markdown("<br>#### Competency Breakdown & Portfolio Objectives", unsafe_allow_html=True)

        for item in gap_result["missing_skill_details"]:
            with st.expander(f"{item['skill'].upper()} ({item['priority']} Priority • Est. {item['estimated_weeks']} wks)"):
                st.markdown(f"**Core Concepts:** {item['concepts']}")
                st.markdown(f"**Recommended Portfolio Deliverable:** {item['project']}")
                st.markdown(f"**Industry Credentials:** {item['certifications']}")
    else:
        st.success(f"Candidate profile satisfies 100% of benchmark competencies for {gap_result['role_name']}.")

    # Phased Milestone Learning Roadmap
    st.markdown("<br>#### Structured 3-Phase Milestone Timeline", unsafe_allow_html=True)

    for p_idx, phase in enumerate(gap_result["roadmap_phases"], 1):
        phase_class = f"phase-{p_idx}" if p_idx <= 3 else "phase-1"
        skills_str = ", ".join(phase["skills"]) if phase["skills"] else "None (Core Stack Fully Satisfied)"

        st.markdown(
            f"""
            <div class="phase-card {phase_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div style="font-family:'Plus Jakarta Sans'; font-weight:700; font-size:1.05rem; color:#f8fafc;">
                        {phase['phase']}
                    </div>
                    <span class="tag-chip chip-neutral">{phase['duration']}</span>
                </div>
                <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:6px;">
                    <b>Focus:</b> {phase['objective']}
                </div>
                <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:6px;">
                    <b>Target Competencies:</b> <span style="color:#60a5fa; font-weight:600;">{skills_str}</span>
                </div>
                <div style="font-size:0.82rem; color:#93c5fd; background:rgba(37,99,235,0.18); padding:6px 12px; border-radius:6px; border:1px solid #2563eb;">
                    <b>Deliverable:</b> {phase['deliverable']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 4: COMPETENCY MATRIX & SECTIONS
# =============================================================================
with tab_skills:
    st.markdown("### Competency Matrix & Structured Document Inspector")
    st.markdown("Inspection of recognized technical skill sets, category distribution, and segmented document blocks.")

    col_radar, col_skill_list = st.columns([1.1, 1.3])

    with col_radar:
        st.markdown("#### Domain Footprint Radar")
        fig_radar = build_skill_category_radar(skills_by_category)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_skill_list:
        st.markdown("#### Verified Technical Competencies")

        # Interactive filter
        search_skill = st.text_input("Filter Skills:", placeholder="e.g. Python, Docker, React")

        for category, cat_skills in skills_by_category.items():
            disp_skills = cat_skills
            if search_skill.strip():
                disp_skills = [s for s in cat_skills if search_skill.lower() in s.lower()]

            if disp_skills:
                chips = " ".join([f"<span class='tag-chip chip-tech'>{s}</span>" for s in disp_skills])
                st.markdown(
                    f"""
                    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:10px 14px; margin-bottom:8px; box-shadow:0 2px 8px rgba(0,0,0,0.3);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-weight:700; font-size:0.85rem; color:#f8fafc;">{category}</span>
                            <span class="tag-chip chip-neutral">{len(disp_skills)} skills</span>
                        </div>
                        <div>{chips}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("#### Parsed Document Sections")

    for sec_name, sec_content in sections.items():
        if sec_content.strip():
            with st.expander(f"Section: {sec_name.upper()} ({len(sec_content.split())} words)"):
                st.code(sec_content, language="text")


# =============================================================================
# TAB 5: STAR BULLET STUDIO
# =============================================================================
with tab_bullet:
    st.markdown("### STAR Method Resume Bullet Point Studio")
    st.markdown(
        "Transform passive or vague bullet points into high-impact, ATS-optimized accomplishment statements using the **STAR** (Situation, Task, Action, Result) methodology."
    )

    preset_bullets = {
        "Custom Bullet Input": "",
        "Example 1: Machine Learning & Modeling": "Worked on machine learning models with Python to classify customer data and helped improve performance.",
        "Example 2: Full Stack & Web Development": "Responsible for developing web application with React and Node.js for client dashboard.",
        "Example 3: Data Analytics & Business Intelligence": "Assisted team in running SQL queries and created weekly spreadsheets in Excel for management.",
        "Example 4: Cloud & DevOps Infrastructure": "Helped with Docker containers and deployed code onto AWS servers.",
    }

    selected_preset = st.selectbox("Load Sample Bullet Point:", list(preset_bullets.keys()))

    initial_text = preset_bullets[selected_preset] if selected_preset != "Custom Bullet Input" else "Worked on machine learning models with Python to classify customer data and helped improve performance."

    user_bullet = st.text_area(
        "Candidate Resume Bullet Point:",
        value=initial_text,
        height=90,
    )

    if user_bullet.strip():
        b_clean = clean_text(user_bullet)
        b_words = b_clean.split()

        # Check action verbs
        import ats_scorer as ats_module
        matched_verbs = [w for w in b_words if w in ats_module.STRONG_ACTION_VERBS]
        has_verbs = len(matched_verbs) > 0

        # Check metrics
        has_metrics = bool(re.search(r"\b(?:\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:k|m|b)?|\d+\+?|\d+x)\b", user_bullet, re.IGNORECASE))

        # Check technical keywords
        extracted_b_skills = skill_extractor.extract_skills(user_bullet)["matched_skills"]

        st.markdown("<br>", unsafe_allow_html=True)
        col_diag1, col_diag2, col_diag3 = st.columns(3)

        with col_diag1:
            if has_verbs:
                st.success(f"Action Verb: {', '.join(matched_verbs[:2])}")
            else:
                st.error("Passive Opening Phrasing Detected")

        with col_diag2:
            if has_metrics:
                st.success("Quantified Metric Present")
            else:
                st.warning("Missing Measurable Metric")

        with col_diag3:
            if extracted_b_skills:
                st.success(f"Keywords: {', '.join(extracted_b_skills[:3])}")
            else:
                st.info("Few Technical Tools Identified")

        st.markdown("---")
        st.markdown("#### High-Impact Executive Alternatives")

        alt1 = "Architected and deployed an end-to-end customer classification pipeline in Python, boosting inference throughput by 35% across 500K daily records."
        alt2 = "Spearheaded predictive customer churn modeling using Scikit-Learn and Pandas, achieving an 89% F1-score and reducing churn by 12% annually."
        alt3 = "Containerized and scaled a machine learning classification service with Docker and FastAPI, reducing API latency from 180ms to 42ms."

        st.markdown(
            f"""
            <div style="background:rgba(37,99,235,0.12); border:1px solid #2563eb; border-radius:10px; padding:14px; margin-bottom:10px;">
                <div style="font-weight:700; color:#60a5fa; font-size:0.85rem; margin-bottom:2px;">Performance & Scale Focus (STAR)</div>
                <div style="color:#f8fafc; font-size:0.92rem;">• {alt1}</div>
            </div>
            <div style="background:rgba(14,165,233,0.12); border:1px solid #0284c7; border-radius:10px; padding:14px; margin-bottom:10px;">
                <div style="font-weight:700; color:#38bdf8; font-size:0.85rem; margin-bottom:2px;">Business ROI & Commercial Outcome Focus (STAR)</div>
                <div style="color:#f8fafc; font-size:0.92rem;">• {alt2}</div>
            </div>
            <div style="background:rgba(99,102,241,0.12); border:1px solid #4f46e5; border-radius:10px; padding:14px; margin-bottom:10px;">
                <div style="font-weight:700; color:#818cf8; font-size:0.85rem; margin-bottom:2px;">Systems Architecture & Engineering Focus (STAR)</div>
                <div style="color:#f8fafc; font-size:0.92rem;">• {alt3}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 6: EXECUTIVE AUDIT REPORT
# =============================================================================
with tab_export:
    st.markdown("### Executive Audit Report & Export")
    st.markdown("Download formatted compliance summaries and matching analytics for internal review.")

    # Generate Markdown Report
    sec_pts = ats_res['breakdown'].get('section_structure', {}).get('score', 0)
    av_pts = ats_res['breakdown'].get('action_verbs', {}).get('score', 0)
    av_found_count = len(ats_res['breakdown'].get('action_verbs', {}).get('details', {}).get('found_verbs', []))
    qi_pts = ats_res['breakdown'].get('quantifiable_impact', {}).get('score', 0)
    qi_metric_count = ats_res['breakdown'].get('quantifiable_impact', {}).get('details', {}).get('metrics_count', 0)
    lh_pts = ats_res['breakdown'].get('length_health', ats_res['breakdown'].get('length_hygiene', {})).get('score', 0)
    ci_pts = ats_res['breakdown'].get('contact_info', {}).get('score', 0)
    sd_pts = ats_res['breakdown'].get('skill_density', {}).get('score', 0)

    report_md = f"""# Executive Resume Audit & Career Benchmark Report
**Document ID:** {resume_data['file_name']}
**Audit Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Total Volume:** {word_count} words
**Document Format:** {resume_data['format']}

---

## 1. Executive Summary
- **Overall ATS Compliance Score:** {ats_res['overall_score']} / 100
- **Classification Rating:** Grade {ats_res['grade']} ({ats_res['tier']})
- **Primary Career Alignment:** {top_role['role_name'] if top_role else 'N/A'} ({top_role['match_score'] if top_role else 0}% Match)
- **Technical Skills Identified:** {len(matched_skills)} verified competencies

---

## 2. ATS Compliance Pillars
- **Section Architecture:** {sec_pts} / 25 pts
- **Action Verb Saturation:** {av_pts} / 20 pts ({av_found_count} verbs)
- **Quantifiable Metrics:** {qi_pts} / 20 pts ({qi_metric_count} metrics)
- **Length & Readability:** {lh_pts} / 15 pts
- **Contact & Web Profiles:** {ci_pts} / 10 pts
- **Technical Keyword Density:** {sd_pts} / 10 pts

### Priority Recommendations:
"""
    for tip in ats_res["suggestions"]:
        report_md += f"- {tip}\n"

    report_md += "\n---\n\n## 3. Top Career Matches Ranking\n"
    for idx, r in enumerate(recommendations, 1):
        report_md += f"### {idx}. {r['role_name']} ({r['category']}) - {r['match_score']}% Match\n"
        report_md += f"- **Skill Coverage:** {r['skill_score']}%\n"
        report_md += f"- **Semantic Relevance:** {r['semantic_score']}%\n"
        report_md += f"- **Matched Skills ({len(r['matched_skills'])}):** {', '.join(r['matched_skills'])}\n"
        report_md += f"- **Missing Skills ({len(r['missing_skills'])}):** {', '.join(r['missing_skills']) if r['missing_skills'] else '100% matched'}\n\n"

    report_md += "---\n\n## 4. Extracted Skills Inventory\n"
    for cat, c_skills in skills_by_category.items():
        report_md += f"**{cat} ({len(c_skills)}):** {', '.join(c_skills)}\n\n"

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="Download Audit Report (Markdown)",
            data=report_md,
            file_name=f"resume_audit_{candidate_name.lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_exp2:
        json_export = {
            "document": resume_data["file_name"],
            "ats_evaluation": ats_res,
            "skills": matched_skills,
            "skills_by_category": skills_by_category,
            "recommendations": recommendations,
        }
        st.download_button(
            label="Download Data Object (JSON)",
            data=json.dumps(json_export, indent=2),
            file_name=f"resume_audit_{candidate_name.lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown(" Document Preview")
    st.text_area("Audit Preview", value=report_md, height=320)
