import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insights import (
    returnBasicCorr,
    returnAggDay,
    brent_lag,
    wti_lag,
    rolling_brent,
    rolling_wti
)

st.set_page_config(page_title="Geopolitical Sentiment & Oil Price Correlation", page_icon="⛽", layout="wide")

# ── CSS — crude oil aesthetic ─────────────────────────────────────────────────
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
    padding: 0.75rem 1.25rem !important;
    overflow-x: auto !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Interpretation panel ── */
.interp-panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
    line-height: 1.75;
}
.interp-panel .interp-label {
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.6rem;
}
.interp-panel .interp-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
    align-items: flex-start;
}
.interp-panel .interp-dot {
    color: var(--amber);
    flex-shrink: 0;
    margin-top: 1px;
}
.interp-panel .interp-muted {
    color: var(--muted);
}

/* ── Legend row ── */
.lag-legend {
    display: flex;
    gap: 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    align-items: center;
}
.lag-legend .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    flex-shrink: 0;
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
    Correlation <span class="accent">Analysis</span>
    <span class="tag">Iran–Israel · Feb–Mar 2025</span>
</div>
<div class="page-caption">
    Pearson &amp; Spearman correlation &nbsp;·&nbsp; lag analysis (0–5 days) &nbsp;·&nbsp;
    significance threshold p &lt; 0.05
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
)

# ── Average Sentiment Score Per Day table ─────────────────────────────────────
st.subheader("Daily Aggregated Data")
agg = returnAggDay()
columns_agg = ["Date", "Sentiment Score", "Brent Close", "WTI Close"]

fig = go.Figure(data=[go.Table(
    columnwidth=[100, 120, 120, 120],
    header=dict(
        values=[f"<b>{c}</b>" for c in columns_agg],
        align="left",
        fill_color="#1a1e2a",
        font=dict(family="DM Mono, monospace", size=11, color="#6b7280"),
        line_color="#242836",
        height=36,
    ),
    cells=dict(
        values=[agg.date, agg.sentiment_score, agg.brent_close, agg.wti_close],
        align="left",
        fill_color=["#13161e", "#13161e", "#13161e", "#13161e"],
        font=dict(family="DM Mono, monospace", size=11, color="#e8eaf0"),
        line_color="#242836",
        height=32,
    )
)])
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Basic Correlation table (Pearson + Spearman for Brent and WTI) ─────────────
# ── Basic Correlation table (Pearson + Spearman for Brent and WTI) ─────────────
st.subheader("Pearson & Spearman Correlation")
corr = returnBasicCorr()
columns_corr = ["Asset", "Pearson", "Spearman", "P-value"]

fig = go.Figure(data=[go.Table(
    columnwidth=[80, 120, 120, 120],
    header=dict(
        values=[f"<b>{c}</b>" for c in columns_corr],
        align="left",
        fill_color="#1a1e2a",
        font=dict(family="DM Mono, monospace", size=11, color="#6b7280"),
        line_color="#242836",
        height=36,
    ),
    cells=dict(
        values=[corr.asset, corr.pearson, corr.spearman, corr.pvalue],
        align="left",
        fill_color=["#13161e", "#13161e", "#13161e", "#13161e", "#13161e"],
        font=dict(family="DM Mono, monospace", size=11, color="#e8eaf0"),
        line_color="#242836",
        height=32,
    )
)])
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  margin=dict(l=0, r=0, t=0, b=0),
                  height=36 + (32 * len(corr)) + 16)   # header + rows + padding
st.plotly_chart(fig, use_container_width=True)


st.divider()


# ── Lag analysis bar charts — Brent vs WTI side by side ──────────────────────
st.subheader("Lag Analysis — Sentiment → Price (0–5 days)")

# Color coding: significant (p < 0.05) = amber, insignificant = muted gray
SIG_COLOR   = "#e89c3f"
INSIG_COLOR = "#2e3748"
SIG_BORDER  = "#e89c3f"
INSIG_BORDER = "#3a4558"

st.markdown("""
<div class="lag-legend">
    <span><span class="dot" style="background:#e89c3f"></span>Significant (p &lt; 0.05)</span>
    <span><span class="dot" style="background:#3a4558;border:1px solid #4a5568"></span>Insignificant</span>
</div>
""", unsafe_allow_html=True)

col_brent, col_wti = st.columns(2)

# ── Brent lag bar chart ───────────────────────────────────────────────────────
with col_brent:
    brent = brent_lag()  # columns: lag_days, pearson_r, p_value

    colors  = [SIG_COLOR   if p < 0.05 else INSIG_COLOR  for p in brent["p_value"]]
    borders = [SIG_BORDER  if p < 0.05 else INSIG_BORDER for p in brent["p_value"]]

    fig_brent = go.Figure()
    fig_brent.add_trace(go.Bar(
        x=[f"Lag {d}" for d in brent["lag_days"]],
        y=brent["pearson_r"],
        marker=dict(
            color=colors,
            line=dict(color=borders, width=1),
        ),
        name="Pearson r",
        hovertemplate="<b>%{x}</b><br>Pearson r: %{y:.3f}<extra></extra>",
    ))
    fig_brent.update_layout(
        **BASE,
        title=dict(text="Brent Crude", font=dict(size=12, color="#8a94a8"), x=0),
        showlegend=False,
    )
    fig_brent.update_yaxes(range=[-0.8, 0.1], title="Pearson r",
                           title_font=dict(size=10, color="#5a6478"))
    # Reference line at 0
    fig_brent.add_hline(y=0, line=dict(color="#3a4558", width=0.8))
    st.plotly_chart(fig_brent, use_container_width=True)

# ── WTI lag bar chart ─────────────────────────────────────────────────────────
with col_wti:
    wti = wti_lag()  # columns: lag_days, pearson_r, p_value

    colors  = [SIG_COLOR   if p < 0.05 else INSIG_COLOR  for p in wti["p_value"]]
    borders = [SIG_BORDER  if p < 0.05 else INSIG_BORDER for p in wti["p_value"]]

    fig_wti = go.Figure()
    fig_wti.add_trace(go.Bar(
        x=[f"Lag {d}" for d in wti["lag_days"]],
        y=wti["pearson_r"],
        marker=dict(
            color=colors,
            line=dict(color=borders, width=1),
        ),
        name="Pearson r",
        hovertemplate="<b>%{x}</b><br>Pearson r: %{y:.3f}<extra></extra>",
    ))
    fig_wti.update_layout(
        **BASE,
        title=dict(text="WTI Crude", font=dict(size=12, color="#8a94a8"), x=0),
        showlegend=False,
    )
    fig_wti.update_yaxes(range=[-0.8, 0.1], title="Pearson r",
                         title_font=dict(size=10, color="#5a6478"))
    fig_wti.add_hline(y=0, line=dict(color="#3a4558", width=0.8))
    st.plotly_chart(fig_wti, use_container_width=True)

st.divider()

# ── Rolling correlation — Brent vs WTI (14-day window) ───────────────────────
st.subheader("14-Day Rolling Correlation — Sentiment vs. Price")

r_brent = rolling_brent()  # columns: index, 0  (index = day offset, 0 = rolling corr value)
r_wti   = rolling_wti()

# Drop leading NaN rows — first 13 days have no complete 14-day window yet
r_brent = r_brent.dropna(subset=[r_brent.columns[1]])
r_wti   = r_wti.dropna(subset=[r_wti.columns[1]])

col_rb, col_rw = st.columns(2)

# ── Brent rolling line chart ──────────────────────────────────────────────────
with col_rb:
    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(
        x=r_brent.iloc[:, 0],
        y=r_brent.iloc[:, 1],
        mode="lines",
        line=dict(color="#e89c3f", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(232,156,63,0.07)",
        hovertemplate="Day %{x}<br>r = %{y:.3f}<extra></extra>",
        name="Brent",
    ))
    fig_rb.add_hline(y=0, line=dict(color="#3a4558", width=0.8))
    fig_rb.update_layout(
        **BASE,
        title=dict(text="Brent Crude", font=dict(size=12, color="#8a94a8"), x=0),
        showlegend=False,
    )
    fig_rb.update_yaxes(range=[-0.85, 0.15], title="Rolling r",
                        title_font=dict(size=10, color="#5a6478"))
    fig_rb.update_xaxes(title="Day", title_font=dict(size=10, color="#5a6478"))
    st.plotly_chart(fig_rb, use_container_width=True)

# ── WTI rolling line chart ────────────────────────────────────────────────────
with col_rw:
    fig_rw = go.Figure()
    fig_rw.add_trace(go.Scatter(
        x=r_wti.iloc[:, 0],
        y=r_wti.iloc[:, 1],
        mode="lines",
        line=dict(color="#c94f4f", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(201,79,79,0.07)",
        hovertemplate="Day %{x}<br>r = %{y:.3f}<extra></extra>",
        name="WTI",
    ))
    fig_rw.add_hline(y=0, line=dict(color="#3a4558", width=0.8))
    fig_rw.update_layout(
        **BASE,
        title=dict(text="WTI Crude", font=dict(size=12, color="#8a94a8"), x=0),
        showlegend=False,
    )
    fig_rw.update_yaxes(range=[-0.85, 0.15], title="Rolling r",
                        title_font=dict(size=10, color="#5a6478"))
    fig_rw.update_xaxes(title="Day", title_font=dict(size=10, color="#5a6478"))
    st.plotly_chart(fig_rw, use_container_width=True)

st.divider()

# ── Plain-English interpretation panel ───────────────────────────────────────
st.markdown("""
<div class="interp-panel">
    <div class="interp-label">Key Findings</div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span><strong>Brent</strong> shows a statistically significant negative correlation at lags 0, 2, 4, and 5,
        suggesting the global benchmark digests geopolitical sentiment
        <strong>over a multi-day window</strong> rather than reacting all at once.</span>
    </div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span><strong>WTI</strong> is only significant at lag 0 — it reacts to sentiment
        <strong>same-day</strong>, with no statistically meaningful delayed effect.</span>
    </div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span class="interp-muted">This contrast is consistent with Brent's role as the
        Middle East-linked global benchmark, making it more sensitive to prolonged
        geopolitical developments than the more domestically-priced WTI.</span>
    </div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span>Both assets show a <strong>stable, consistently negative rolling correlation</strong>
        across the study period — this is not noise. The relationship between negative sentiment
        and rising prices holds across the full 28-day window.</span>
    </div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span>The <strong>strongest inverse relationship</strong> for both assets occurs around
        days 19–20 (March 18–20), where rolling r drops to approximately <strong>−0.67 to −0.69</strong>,
        indicating peak market sensitivity to geopolitical developments during that period.</span>
    </div>
    <div class="interp-row">
        <span class="interp-dot">—</span>
        <span class="interp-muted">WTI's rolling correlation strengthens and <em>stays</em> strong
        from day 19 onward (r ≈ −0.57 to −0.69), while Brent's weakens slightly toward the end
        of the window — consistent with WTI reacting sharply and holding, and Brent absorbing
        the signal more gradually.</span>
    </div>
</div>
""", unsafe_allow_html=True)