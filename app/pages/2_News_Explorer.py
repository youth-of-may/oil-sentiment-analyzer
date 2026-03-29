"""
dashboard.py — Overview dashboard for Geopolitical Sentiment & Oil Price Correlation.
Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from insights import (
    newsPerDate
)

st.set_page_config(
    page_title="Geopolitical Sentiment & Oil Price Correlation",
    page_icon="⛽",
    layout="wide"
)

# ── CSS — crude oil aesthetic: deep navy, amber, petroleum green ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

:root {
    --bg:         #0b0e13;
    --surface:    #12161f;
    --surface-2:  #181d28;
    --border:     #232a38;
    --amber:      #e89c3f;
    --amber-dim:  rgba(232,156,63,0.12);
    --green:      #3a9e6e;
    --red:        #c94f4f;
    --text:       #d8dce8;
    --muted:      #5a6478;
    --font-head:  'Syne', sans-serif;
    --font-mono:  'DM Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(232,156,63,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,156,63,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] { background-color: var(--surface) !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.main .block-container {
    padding: 2.5rem 3rem 3rem !important;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* ── Header ── */
.page-title {
    font-family: var(--font-head);
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.1;
}
.page-title .accent { color: var(--amber); }
.page-caption {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.4rem;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
}
.tag {
    display: inline-block;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--amber);
    background: var(--amber-dim);
    border: 1px solid rgba(232,156,63,0.22);
    padding: 2px 9px 3px;
    border-radius: 3px;
    margin-left: 0.7rem;
    vertical-align: middle;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--amber) !important;
    border-radius: 6px !important;
    padding: 1.1rem 1.4rem !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] > div {
    font-family: var(--font-head) !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Section subheaders ── */
h3, .stSubheader {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.6rem !important;
}

/* ── Chart containers ── */
[data-testid="stPlotlyChart"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.75rem !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Date out-of-range banner ── */
.date-warning {
    background: rgba(232,156,63,0.07);
    border: 1px solid rgba(232,156,63,0.28);
    border-left: 3px solid #e89c3f;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #e89c3f;
    margin-top: 0.75rem;
}
.date-warning .warn-label {
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.3rem;
}

/* ── No-articles empty state ── */
.empty-state {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 6px;
    padding: 3rem 2rem;
    text-align: center;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.7;
    margin-top: 0.75rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    Geopolitical Sentiment &amp; <span class="accent">Oil Price</span>
    <span class="tag">Iran–Israel · Feb–Mar 2025</span>
</div>
<div class="page-caption">
    NewsAPI headlines &nbsp;·&nbsp; yfinance prices &nbsp;·&nbsp;
    Groq LLaMA 3.1 sentiment &nbsp;·&nbsp; scipy correlation analysis
</div>
""", unsafe_allow_html=True)

# ── Shared Plotly base layout ─────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#5a6478", size=11),
    margin=dict(l=16, r=16, t=32, b=16),
    legend=dict(
        orientation="h", y=1.08, x=0,
        font=dict(size=10, color="#8a94a8"),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="#1e2432", linecolor="#232a38", zeroline=False,
        tickfont=dict(color="#5a6478", size=10),
    ),
    yaxis=dict(
        gridcolor="#1e2432", linecolor="#232a38", zeroline=False,
        tickfont=dict(color="#5a6478", size=10),
    ),
    yaxis2=dict(
        gridcolor="rgba(0,0,0,0)", linecolor="#232a38", zeroline=False,
        tickfont=dict(color="#5a6478", size=10),
        overlaying="y", side="right",
    ),
)

# ── Metrics row ───────────────────────────────────────────────────────────────
st.write("Daily news coverage scored by sentiment · select a date to explore")

st.divider()

# ── Date range constants ──────────────────────────────────────────────────────
DATE_MIN = date(2026, 2, 28)
DATE_MAX = date(2026, 3, 27)

# ── Header row + date picker ──────────────────────────────────────────────────
col_head, col_select = st.columns([3, 1])

with col_head:
    st.subheader("Articles by Date")

with col_select:
    selected_date = st.date_input("Select Date", value=DATE_MIN)

# ── Date validation: out-of-range guard ──────────────────────────────────────
if selected_date < DATE_MIN or selected_date > DATE_MAX:
    st.markdown(f"""
    <div class="date-warning">
        <div class="warn-label">Out of range</div>
        No data for <strong>{selected_date.strftime("%b %d, %Y")}</strong>.
        Dataset covers <strong>Feb 28 – Mar 27, 2025</strong> only.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load articles for the selected date ──────────────────────────────────────
news = newsPerDate(selected_date)

# ── Empty state: valid date but no articles collected that day ────────────────
if news.empty:
    st.markdown(f"""
    <div class="empty-state">
        — no articles found for {selected_date.strftime("%b %d, %Y")} —<br>
        <span style="opacity:0.5;font-size:0.72rem">
            This may be a weekend or a day with no relevant coverage in the dataset.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Articles table ────────────────────────────────────────────────────────────
columns = ["Title", "Description", "Date", "Sentiment Score"]

fig = go.Figure(data=[go.Table(
    columnwidth=[90, 420, 90],
    header=dict(
        values=[f"<b>{c}</b>" for c in columns],
        align="left",
        fill_color="#1a1e2a",
        font=dict(family="DM Mono, monospace", size=11, color="#6b7280"),
        line_color="#242836",
        height=36,
    ),
    cells=dict(
        values=[news.title, news.description, news.publishedAt, news.sentiment_score],
        align="left",
        fill_color=["#13161e", "#13161e", "#13161e"],
        font=dict(family="DM Mono, monospace", size=11, color="#e8eaf0"),
        line_color="#242836",
        height=32,
    )
)])
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig, use_container_width=True)