"""
dashboard.py — Overview dashboard for Geopolitical Sentiment & Oil Price Correlation.
Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from insights import (
    count_articles,
    average_sentiment,
    brent_sentimentPrice,
    wti_sentimentPrice,
    returnTopNNews,
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
col1, col2, col3 = st.columns(3)
col1.metric("Date Range", "Feb 28 – Mar 27")
col2.metric("Total Articles", count_articles())
col3.metric("Avg Daily Sentiment", average_sentiment())

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
brent = brent_sentimentPrice()   # columns: date, sentiment_score, brent_close
wti   = wti_sentimentPrice()     # columns: date, sentiment_score, wti_close

# ── Brent: dual-axis chart ────────────────────────────────────────────────────
st.subheader("Brent crude — sentiment vs. closing price")

fig_brent = make_subplots(specs=[[{"secondary_y": True}]])

fig_brent.add_trace(
    go.Scatter(
        x=brent["date"],
        y=brent["sentiment_score"],
        name="Sentiment score",
        mode="lines",
        line=dict(color="#e83f3f", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(232, 63, 63, 0.1)",
    ),
    secondary_y=False,
)
fig_brent.add_trace(
    go.Scatter(
        x=brent["date"],
        y=brent["brent_close"],
        name="Brent close (USD)",
        mode="lines",
        line=dict(color="#ecb638", width=1.8),
    ),
    secondary_y=True,
)

fig_brent.update_layout(**BASE)
fig_brent.update_yaxes(title_text="Sentiment score", secondary_y=False,
                       title_font=dict(size=10, color="#5a6478"))
fig_brent.update_yaxes(title_text="Brent close (USD)", secondary_y=True,
                       title_font=dict(size=10, color="#5a6478"))

st.plotly_chart(fig_brent, use_container_width=True)

st.divider()

# ── WTI: dual-axis chart ──────────────────────────────────────────────────────
st.subheader("WTI crude — sentiment vs. closing price")

fig_wti = make_subplots(specs=[[{"secondary_y": True}]])

fig_wti.add_trace(
    go.Scatter(
        x=wti["date"],
        y=wti["sentiment_score"],
        name="Sentiment score",
        mode="lines",
        line=dict(color="#e83f3f", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(232, 63, 63, 0.1)",
    ),
    secondary_y=False,
)
fig_wti.add_trace(
    go.Scatter(
        x=wti["date"],
        y=wti["wti_close"],
        name="WTI close (USD)",
        mode="lines",
        line=dict(color="#7f6cff", width=1.8),
    ),
    secondary_y=True,
)

fig_wti.update_layout(**BASE)
fig_wti.update_yaxes(title_text="Sentiment score", secondary_y=False,
                     title_font=dict(size=10, color="#5a6478"))
fig_wti.update_yaxes(title_text="WTI close (USD)", secondary_y=True,
                     title_font=dict(size=10, color="#5a6478"))

st.plotly_chart(fig_wti, use_container_width=True)

st.divider()

# ── Top Articles with Negative Sentiment Scores ──────────────────────────────
col_head, col_select = st.columns([3, 1])

with col_head:
    st.subheader("Top Articles with the Lowest Sentiment Scores")


with col_select:
    top_n = st.selectbox(
        "Show top",
        options=[5, 10, 15, 20],
        index=0,
        label_visibility="collapsed",
    )

top_news = returnTopNNews(top_n)
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
        values=[top_news.title, top_news.description, top_news.publishedAt, top_news.sentiment_score],
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