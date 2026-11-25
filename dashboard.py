"""
BookGuard AI – Travel Booking Anomaly Detection Dashboard
Streamlit dashboard integrating with the existing FastAPI /score endpoint.
Model, preprocessing, and CSV artefacts are NOT modified here.
"""

import json
import io
import math
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLING – dark navy + teal/cyan palette
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Root palette ── */
    :root {
        --bg-primary:   #0b1120;
        --bg-secondary: #0f1a2e;
        --bg-card:      #131f35;
        --bg-card2:     #162540;
        --teal:         #00c8b4;
        --cyan:         #00d4ff;
        --text-primary: #e8f0fe;
        --text-muted:   #8ba0bb;
        --border:       #1e3250;
        --danger:       #ff4f72;
        --warn:         #ffa040;
        --ok:           #28e0a4;
    }

    /* ── App background ── */
    .stApp { background-color: var(--bg-primary); color: var(--text-primary); }
    section[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; }

    /* ── Hide default header / footer ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── KPI cards ── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label { font-size: 0.75rem; color: var(--text-muted); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 6px; }
    .kpi-value { font-size: 2.0rem; font-weight: 700; color: var(--teal); line-height: 1; }
    .kpi-value.danger { color: var(--danger); }
    .kpi-value.warn   { color: var(--warn); }
    .kpi-value.ok     { color: var(--ok); }
    .kpi-sub  { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: var(--cyan); border-bottom: 1px solid var(--border);
        padding-bottom: 6px; margin: 18px 0 12px 0;
    }

    /* ── Score result box ── */
    .score-box {
        background: var(--bg-card2);
        border-radius: 12px;
        border: 1px solid var(--border);
        padding: 24px 28px;
        margin-top: 16px;
    }
    .score-big { font-size: 2.4rem; font-weight: 700; }
    .score-flag   { color: var(--danger); }
    .score-normal { color: var(--ok); }

    /* ── Disclaimer box ── */
    .disclaimer {
        background: rgba(255,160,64,.08);
        border-left: 3px solid var(--warn);
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.78rem;
        color: #c8a060;
        margin-top: 10px;
    }

    /* ── Sidebar navigation: high-contrast labels and selected state ── */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        background: #111c30 !important;
        border: 1px solid #263b57 !important;
        border-radius: 8px !important;
        padding: 0.55rem 0.7rem !important;
        color: #ffffff !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: #1a3048 !important;
        border-color: #00c8b4 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has([aria-checked="true"]) {
        background: #008f83 !important;
        border-color: #00d4c0 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has([aria-checked="true"]) * {
        color: #ffffff !important;
    }

    /* ── Sidebar brand ── */
    .brand-title {
        font-size: 1.35rem; font-weight: 800;
        background: linear-gradient(90deg, var(--teal), var(--cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: .04em;
    }
    .brand-sub { font-size: 0.72rem; color: var(--text-muted); letter-spacing: .05em; }

    /* ── Dataframe overrides ── */
    .stDataFrame thead th { background: #eef1f5 !important; color: #586273 !important; }
    .stDataFrame tbody td { background: #13243d !important; color: #e8eef7 !important; }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    /* ── Global input label override ── */
    label { color: var(--text-muted) !important; font-size: 0.82rem !important; }

    /* ── Sidebar text input (API URL) ── */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #131f35 !important;
        color: #ffffff !important;
        border: 1px solid #2b4565 !important;
        border-radius: 9px !important;
        caret-color: #00d4ff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
        border-color: #00c8b4 !important;
        box-shadow: 0 0 0 1px #00c8b4 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
        color: #8ba0bb !important;
    }

    /* ── API connection status banner ── */
    .api-status {
        background: rgba(0, 200, 180, 0.10);
        color: #00e0c2;
        border: 1px solid rgba(0, 200, 180, 0.35);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 8px 0 16px 0;
    }

    /* ═══════════════════════════════════════════════════════════
       BOOKING DETECTION PAGE – dark navy number inputs
       ═══════════════════════════════════════════════════════════ */

    /* Number input container background */
    [data-testid="stNumberInput"] > div {
        background-color: #0f1a2e !important;
        border: 1px solid #1e3a5c !important;
        border-radius: 8px !important;
    }

    /* The actual <input> text field */
    [data-testid="stNumberInput"] input {
        background-color: #0f1a2e !important;
        color: #e8f0fe !important;
        caret-color: #00d4ff !important;
        border: none !important;
        font-size: 0.92rem !important;
    }

    /* Focus ring */
    [data-testid="stNumberInput"] > div:focus-within {
        border-color: #00c8b4 !important;
        box-shadow: 0 0 0 1.5px rgba(0,200,180,0.35) !important;
    }

    /* Step buttons (± arrows) */
    [data-testid="stNumberInput"] button {
        background-color: #131f35 !important;
        color: #00c8b4 !important;
        border: none !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background-color: #1a2e4a !important;
        color: #00d4ff !important;
    }

    /* Number input labels – make them clearly visible */
    [data-testid="stNumberInput"] label {
        color: #8ba0bb !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────
DATA_DIR = "."

@st.cache_data(show_spinner=False)
def load_model_config():
    with open(f"{DATA_DIR}/model_config.json") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_run_summary():
    with open(f"{DATA_DIR}/run_summary.json") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_anomaly_results():
    df = pd.read_csv(
        f"{DATA_DIR}/booking_anomaly_results.csv",
        parse_dates=["date_time", "booking_date"],
    )
    return df

@st.cache_data(show_spinner=False)
def load_feature_comparison():
    return pd.read_csv(f"{DATA_DIR}/feature_comparison.csv")

@st.cache_data(show_spinner=False)
def load_flagged_market():
    return pd.read_csv(f"{DATA_DIR}/flagged_market_analysis.csv")

@st.cache_data(show_spinner=False)
def load_flagged_user():
    return pd.read_csv(f"{DATA_DIR}/flagged_user_analysis.csv")

@st.cache_data(show_spinner=False)
def load_threshold_comparison():
    return pd.read_csv(f"{DATA_DIR}/threshold_comparison.csv")

# ─────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────
cfg       = load_model_config()
summary   = load_run_summary()
FEATURES  = cfg["features"]          # exact order from model_config.json
THRESHOLD = float(cfg["threshold"])
INPUT_DIM = int(cfg["input_dim"])

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────────────────────
PLOT_BG  = "#0b1120"
PAPER_BG = "#0b1120"
FONT_CLR = "#8ba0bb"
GRID_CLR = "#1e3250"
TEAL     = "#00c8b4"
CYAN     = "#00d4ff"
DANGER   = "#ff4f72"
WARN     = "#ffa040"
OK       = "#28e0a4"

def base_layout(**kwargs):
    defaults = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
    )
    defaults.update(kwargs)
    return defaults


def kpi(label, value, sub="", color_class=""):
    st.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {color_class}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def style_table(frame):
    """Apply the dark-navy table style from the user's reference screenshot."""
    return (
        frame.style
        .set_table_styles([
            {"selector": "table", "props": [
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
                ("width", "100%"),
                ("font-size", "0.84rem"),
                ("border", "1px solid #263b57"),
                ("border-radius", "7px"),
                ("overflow", "hidden"),
            ]},
            {"selector": "thead th", "props": [
                ("background-color", "#eef1f5 !important"),
                ("color", "#586273 !important"),
                ("font-weight", "600"),
                ("border", "1px solid #d8dee8"),
                ("padding", "9px 10px"),
                ("text-align", "left"),
                ("white-space", "nowrap"),
            ]},
            {"selector": "tbody td", "props": [
                ("background-color", "#13243d !important"),
                ("color", "#e8eef7 !important"),
                ("border", "1px solid #203653"),
                ("padding", "8px 10px"),
            ]},
            {"selector": "tbody tr:nth-child(even) td", "props": [
                ("background-color", "#172b46 !important"),
            ]},
        ])
        .set_properties(**{
            "background-color": "#13243d",
            "color": "#e8eef7",
            "border-color": "#203653",
        })
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-title">🛡️ BookGuard AI</div>'
        '<div class="brand-sub">TRAVEL BOOKING ANOMALY DETECTION</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔍 Booking Detection",
            "📋 Anomaly Records",
            "📈 Analytics & Insights",
            "🔬 Fraud Investigation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**🌐 API Connection**")
    api_base = st.text_input(
        "API Base URL", value="http://localhost:8000", label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**🧠 Model Information**")
    st.markdown(
        f"""
        <div style="font-size:0.80rem; color:#8ba0bb; line-height:2.0;">
        <b style="color:#00c8b4">Model</b><br>Autoencoder<br>
        <b style="color:#00c8b4">Input Features</b><br>{INPUT_DIM}<br>
        <b style="color:#00c8b4">Detection Threshold</b><br>{THRESHOLD:.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")


# ─────────────────────────────────────────────────────────────
# PAGE 1 – OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "📊 Overview":

    st.markdown('<h2 style="color:#00d4ff;margin-bottom:4px;">Overview</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<span style="color:#8ba0bb;font-size:0.82rem;">'
        f'Future test period · {summary["future_test_rows"]:,} bookings evaluated after '
        f'chronological cutoff {summary["chronological_cutoff"][:10]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    df      = load_anomaly_results()
    total   = len(df)
    flagged = int(df["is_anomaly"].sum())
    normal  = total - flagged
    rate    = flagged / total * 100

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total Bookings Screened", f"{total:,}", "future test set")
    with c2: kpi("Flagged for Review",      f"{flagged:,}", f"{rate:.2f}% of total", "danger")
    with c3: kpi("Normal Bookings",         f"{normal:,}", "passed screening", "ok")
    with c4: kpi("Anomaly Rate",            f"{rate:.2f}%", f"threshold: {THRESHOLD:.4f}", "warn")

    st.markdown("")

    # ── Row 1: Risk distribution | Score distribution ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        color_map = {"Typical": TEAL, "Review": DANGER}
        fig_pie = px.pie(
            risk_counts, names="risk_level", values="count",
            color="risk_level", color_discrete_map=color_map,
            hole=0.55,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=13, color=FONT_CLR),
            marker=dict(line=dict(color=PLOT_BG, width=3)),
        )
        fig_pie.update_layout(
            **base_layout(height=320, showlegend=True, margin=dict(l=20, r=20, t=20, b=20))
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        cap = df["anomaly_score"].quantile(0.99)
        df_plot = df[df["anomaly_score"] <= cap].copy()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_plot[df_plot["is_anomaly"] == 0]["anomaly_score"],
            name="Normal", nbinsx=50,
            marker_color=TEAL, opacity=0.75,
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_plot[df_plot["is_anomaly"] == 1]["anomaly_score"],
            name="Flagged", nbinsx=50,
            marker_color=DANGER, opacity=0.85,
        ))
        fig_hist.add_vline(
            x=THRESHOLD, line_dash="dash", line_color=WARN, line_width=2,
            annotation_text=f"Threshold {THRESHOLD:.3f}",
            annotation_font_color=WARN,
            annotation_position="top right",
        )
        fig_hist.update_layout(
            **base_layout(height=320, barmode="overlay"),
            xaxis_title="Anomaly Score (MSE)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Flagged bookings over time ──
    st.markdown('<div class="section-header">Flagged Bookings Over Time</div>', unsafe_allow_html=True)

    df_flagged = df[df["is_anomaly"] == 1].copy()
    df_flagged["week"] = df_flagged["date_time"].dt.to_period("W").dt.start_time
    weekly = df_flagged.groupby("week").size().reset_index(name="flagged_count")

    df["week"] = df["date_time"].dt.to_period("W").dt.start_time
    weekly_all = df.groupby("week").size().reset_index(name="total_count")
    weekly_merged = weekly_all.merge(weekly, on="week", how="left").fillna(0)
    weekly_merged["flagged_count"] = weekly_merged["flagged_count"].astype(int)

    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=weekly_merged["week"], y=weekly_merged["total_count"],
        name="Total Bookings", marker_color=TEAL, opacity=0.4,
    ))
    fig_time.add_trace(go.Bar(
        x=weekly_merged["week"], y=weekly_merged["flagged_count"],
        name="Flagged", marker_color=DANGER, opacity=0.9,
    ))
    fig_time.update_layout(
        **base_layout(height=280, barmode="overlay"),
        xaxis_title="Week", yaxis_title="Bookings",
    )
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

    # ── Recently flagged bookings table ──
    st.markdown('<div class="section-header">Recently Flagged Bookings</div>', unsafe_allow_html=True)
    recent_cols = [
        "date_time", "booking_id", "user_id", "hotel_market",
        "amount", "num_guests", "num_rooms", "anomaly_score", "risk_level",
    ]
    recent = (
        df[df["is_anomaly"] == 1][recent_cols]
        .sort_values("date_time", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    recent["anomaly_score"] = recent["anomaly_score"].round(4)
    recent["amount"] = recent["amount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(
        style_table(recent).applymap(
            lambda v: "color: #c62845; font-weight:600" if v == "Review" else "",
            subset=["risk_level"],
        ),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────
# PAGE 2 – BOOKING DETECTION (live /score endpoint)
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Booking Detection":

    st.markdown('<h2 style="color:#00d4ff;margin-bottom:4px;">Live Booking Detection</h2>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#8ba0bb;font-size:0.82rem;">'
        'Enter booking feature values and send them to the FastAPI /score endpoint.</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    # ── API status indicator ──
    with st.spinner("Checking API…"):
        try:
            resp = requests.get(f"{api_base}/", timeout=3)
            if resp.status_code == 200:
                api_ok  = True
                api_msg = "✅ API connected"
            else:
                api_ok  = False
                api_msg = f"⚠️ API returned status {resp.status_code}"
        except Exception:
            api_ok  = False
            api_msg = "❌ API unreachable · start FastAPI first"

    if api_ok:
        st.markdown(
            '<div class="api-status">✓ API connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(api_msg)
        st.info("Run:  `uvicorn app:app --reload`  then refresh this page.")

    st.markdown('<div class="section-header">Feature Inputs</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span style="color:#8ba0bb;font-size:0.76rem;">'
        f'Enter all {INPUT_DIM} features in the required order (matching model_config.json).</span>',
        unsafe_allow_html=True,
    )

    # Feature metadata: (label, min, max, default, step, format, help)
    feature_meta = {
        "num_guests": (
            "Number of Guests", 1.0, 20.0, 2.0, 1.0, "%.0f",
            "Total adult + child guests",
        ),
        "num_rooms": (
            "Number of Rooms", 1.0, 10.0, 1.0, 1.0, "%.0f",
            "Rooms booked",
        ),
        "amount": (
            "Booking Amount (currency)", 0.0, 500000.0, 10000.0, 100.0, "%.2f",
            "Total payment amount",
        ),
        "lead_time_days": (
            "Lead Time (days)", -10.0, 730.0, 30.0, 1.0, "%.1f",
            "Days between booking date and check-in",
        ),
        "stay_duration_days": (
            "Stay Duration (days)", 1.0, 60.0, 3.0, 0.5, "%.1f",
            "Number of nights",
        ),
        "past_user_bookings": (
            "Past User Bookings", 0.0, 50.0, 0.0, 1.0, "%.0f",
            "Lifetime bookings by this user",
        ),
        "past_user_cancellations": (
            "Past User Cancellations", 0.0, 50.0, 0.0, 1.0, "%.0f",
            "Lifetime cancellations by this user",
        ),
        "past_user_cancellation_rate": (
            "User Cancellation Rate", 0.0, 1.0, 0.0, 0.01, "%.4f",
            "Fraction of past bookings cancelled (0.0 – 1.0)",
        ),
        "past_market_bookings": (
            "Past Market Bookings", 0.0, 1000.0, 25.0, 1.0, "%.0f",
            "Historical bookings in this travel market",
        ),
        "past_market_cancellations": (
            "Past Market Cancellations", 0.0, 500.0, 4.0, 1.0, "%.0f",
            "Historical cancellations in this market",
        ),
        "past_market_cancellation_rate": (
            "Market Cancellation Rate", 0.0, 1.0, 0.12, 0.01, "%.4f",
            "Market-level historical cancellation rate (0.0 – 1.0)",
        ),
        "past_user_bookings_7d": (
            "User Bookings – Last 7 Days", 0.0, 20.0, 0.0, 1.0, "%.0f",
            "Recent user activity in the past 7 days",
        ),
    }

    vals = {}
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, feat in enumerate(FEATURES):
        label, mn, mx, default, step, fmt, tip = feature_meta[feat]
        with cols[i % 3]:
            vals[feat] = st.number_input(
                label,
                min_value=mn,
                max_value=mx,
                value=default,
                step=step,
                format=fmt,
                help=tip,
                key=f"feat_{feat}",
            )

    st.markdown("")
    run_btn = st.button("🔎 Run Anomaly Check", type="primary", use_container_width=False)

    if run_btn:
        feature_values = [float(vals[f]) for f in FEATURES]

        if not all(np.isfinite(v) for v in feature_values):
            st.error("All feature values must be finite numbers.")
        elif not api_ok:
            st.error("API is not reachable. Please start the FastAPI server first.")
        else:
            with st.spinner("Sending to /score endpoint…"):
                try:
                    payload = {"values": feature_values}
                    r = requests.post(
                        f"{api_base}/score",
                        json=payload,
                        timeout=10,
                    )
                    if r.status_code == 200:
                        result      = r.json()
                        score_val   = float(result.get("anomaly_score", 0))
                        threshold   = float(result.get("threshold", THRESHOLD))
                        flagged_b   = bool(result.get("flag_for_review", False))
                        note        = result.get("note", "")

                        flag_cls    = "score-flag" if flagged_b else "score-normal"
                        flag_txt    = "🚨 FLAGGED FOR REVIEW" if flagged_b else "✅ NORMAL BOOKING"
                        score_color = DANGER if flagged_b else OK

                        st.markdown(
                            f"""<div class="score-box">
                              <div style="color:#8ba0bb;font-size:0.8rem;margin-bottom:4px;">ANOMALY SCORE</div>
                              <div class="score-big {flag_cls}">{score_val:.6f}</div>
                              <div style="margin-top:12px;font-size:1.1rem;font-weight:700;color:{'#ff4f72' if flagged_b else '#28e0a4'};">{flag_txt}</div>
                              <div style="margin-top:10px;font-size:0.8rem;color:#8ba0bb;">
                                Model Threshold: <b style="color:#ffa040">{threshold:.6f}</b> &nbsp;|&nbsp;
                                Score vs Threshold: <b style="color:{score_color}">{score_val:.6f} {'>' if flagged_b else '<'} {threshold:.6f}</b>
                              </div>
                              <div style="margin-top:10px;font-size:0.76rem;color:#8ba0bb;font-style:italic;">📌 {note}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # Gauge chart
                        cap_gauge = max(threshold * 3, score_val * 1.1)
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_val,
                            number={"valueformat": ".5f", "font": {"color": score_color, "size": 28}},
                            gauge={
                                "axis": {"range": [0, cap_gauge], "tickcolor": FONT_CLR},
                                "bar":  {"color": score_color, "thickness": 0.3},
                                "bgcolor": "#0f1a2e",
                                "bordercolor": GRID_CLR,
                                "steps": [
                                    {"range": [0, threshold],  "color": "#0d2a1f"},
                                    {"range": [threshold, cap_gauge], "color": "#2a0d12"},
                                ],
                                "threshold": {
                                    "line": {"color": WARN, "width": 3},
                                    "thickness": 0.85,
                                    "value": threshold,
                                },
                            },
                            title={"text": "Anomaly Score Gauge", "font": {"color": FONT_CLR, "size": 13}},
                        ))
                        fig_gauge.update_layout(
                            paper_bgcolor=PAPER_BG,
                            font_color=FONT_CLR,
                            height=270,
                            margin=dict(l=30, r=30, t=50, b=10),
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                        with st.expander("📄 Full API Response (JSON)"):
                            st.json(result)

                    elif r.status_code == 400:
                        st.error(f"API validation error (400): {r.json().get('detail', r.text)}")
                    elif r.status_code == 422:
                        st.error(f"Unprocessable payload (422): {r.json()}")
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the API. "
                        "Make sure FastAPI is running (`uvicorn app:app --reload`)."
                    )
                except requests.exceptions.Timeout:
                    st.error("API request timed out after 10 seconds.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")


# ─────────────────────────────────────────────────────────────
# PAGE 3 – ANOMALY RECORDS
# ─────────────────────────────────────────────────────────────
elif page == "📋 Anomaly Records":

    st.markdown('<h2 style="color:#00d4ff;margin-bottom:4px;">Anomaly Records</h2>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#8ba0bb;font-size:0.82rem;">'
        'Browse, filter and export the full future-test evaluation results.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    df = load_anomaly_results()

    # ── Exact bounds derived from the data ──
    # Use floor/ceil to 4 d.p. so that the actual min and max scores
    # are always inside the slider range (plain round() causes boundary rows
    # to be excluded, producing 1,597 instead of 1,599).
    score_min_raw = float(df["anomaly_score"].min())
    score_max_raw = float(df["anomaly_score"].max())
    score_min_dp4 = math.floor(score_min_raw * 10000) / 10000   # floor → never above actual min
    score_max_dp4 = math.ceil(score_max_raw  * 10000) / 10000   # ceil  → never below actual max

    date_min = df["date_time"].min().date()
    date_max = df["date_time"].max().date()

    # ── Filters ──
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        risk_opts = ["All"] + sorted(df["risk_level"].unique().tolist())
        sel_risk  = st.selectbox("Risk Level", risk_opts)
    with fc2:
        status_opts = ["All", "Flagged (is_anomaly=1)", "Normal (is_anomaly=0)"]
        sel_status  = st.selectbox("Anomaly Status", status_opts)
    with fc3:
        sel_date_start = st.date_input(
            "From Date", value=date_min, min_value=date_min, max_value=date_max
        )
    with fc4:
        sel_date_end = st.date_input(
            "To Date", value=date_max, min_value=date_min, max_value=date_max
        )

    # Score range slider – uses floor/ceil bounds so all 1,599 rows pass by default
    sc1, _sc2 = st.columns(2)
    with sc1:
        sel_score = st.slider(
            "Anomaly Score Range",
            min_value=score_min_dp4,
            max_value=score_max_dp4,
            value=(score_min_dp4, score_max_dp4),
        )

    # ── Apply filters ──
    mask = pd.Series(True, index=df.index)

    if sel_risk != "All":
        mask &= df["risk_level"] == sel_risk
    if sel_status == "Flagged (is_anomaly=1)":
        mask &= df["is_anomaly"] == 1
    elif sel_status == "Normal (is_anomaly=0)":
        mask &= df["is_anomaly"] == 0

    mask &= df["date_time"].dt.date >= sel_date_start
    mask &= df["date_time"].dt.date <= sel_date_end
    mask &= df["anomaly_score"] >= sel_score[0]
    mask &= df["anomaly_score"] <= sel_score[1]

    df_filt = df[mask].copy().reset_index(drop=True)

    # ── Filtered KPIs ──
    st.markdown("")
    k1, k2, k3, k4 = st.columns(4)
    filt_total   = len(df_filt)
    filt_flagged = int(df_filt["is_anomaly"].sum())
    filt_normal  = filt_total - filt_flagged
    filt_rate    = filt_flagged / filt_total * 100 if filt_total else 0
    avg_score    = df_filt["anomaly_score"].mean() if filt_total else 0.0

    with k1: kpi("Filtered Records",  f"{filt_total:,}", "matching criteria")
    with k2: kpi("Flagged",           f"{filt_flagged:,}", f"{filt_rate:.1f}% of filtered", "danger")
    with k3: kpi("Normal",            f"{filt_normal:,}", "passed screening", "ok")
    with k4: kpi("Avg Anomaly Score", f"{avg_score:.4f}", f"threshold: {THRESHOLD:.4f}", "warn")

    st.markdown("")

    # ── Records table ──
    st.markdown('<div class="section-header">Records Table</div>', unsafe_allow_html=True)

    display_cols = [
        "date_time", "booking_id", "user_id", "hotel_market",
        "num_guests", "num_rooms", "amount", "lead_time_days", "stay_duration_days",
        "past_user_cancellation_rate", "past_market_cancellation_rate",
        "anomaly_score", "is_anomaly", "risk_level",
    ]
    display_cols = [c for c in display_cols if c in df_filt.columns]

    df_display = df_filt[display_cols].copy()
    df_display["anomaly_score"] = df_display["anomaly_score"].round(6)
    df_display["amount"]        = df_display["amount"].round(2)

    st.dataframe(
        style_table(df_display).applymap(
            lambda v: "color: #c62845; font-weight:600" if v == "Review" else "",
            subset=["risk_level"],
        ).applymap(
            lambda v: "color: #c62845; font-weight:600" if v == 1 else "color: #087f5b",
            subset=["is_anomaly"],
        ),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    # ── CSV download ──
    st.markdown("")
    csv_buf = io.StringIO()
    df_filt.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="bookguard_filtered_records.csv",
        mime="text/csv",
        type="primary",
    )


# ─────────────────────────────────────────────────────────────
# PAGE 4 – ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────────────────────
elif page == "📈 Analytics & Insights":

    st.markdown('<h2 style="color:#00d4ff;margin-bottom:4px;">Analytics & Insights</h2>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#8ba0bb;font-size:0.82rem;">'
        'Precomputed analysis derived from the ML evaluation run.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    df         = load_anomaly_results()
    df_market  = load_flagged_market()
    df_thresh  = load_threshold_comparison()

    # ── Row 1: Market analysis | Threshold comparison ──
    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Renamed from "Top Hotel Markets" to "Top Travel Markets"
        # (uses the actual hotel_market column in the dataset – column unchanged)
        st.markdown(
            '<div class="section-header">Top Travel Markets by Flagged Bookings</div>',
            unsafe_allow_html=True,
        )
        top_markets = df_market.sort_values("flagged_bookings", ascending=False).head(15)
        fig_mkt = px.bar(
            top_markets,
            x="hotel_market",
            y="flagged_bookings",
            color="mean_score",
            color_continuous_scale=[[0, TEAL], [0.5, WARN], [1, DANGER]],
            labels={
                "hotel_market":    "Travel Market ID",
                "flagged_bookings": "Flagged Bookings",
                "mean_score":      "Mean Anomaly Score",
            },
            text="flagged_bookings",
        )
        fig_mkt.update_traces(textposition="outside", textfont_color=FONT_CLR)
        fig_mkt.update_layout(
            **base_layout(
                height=350,
                coloraxis_colorbar=dict(title="Mean Anomaly Score", tickfont_color=FONT_CLR),
            )
        )
        fig_mkt.update_xaxes(type="category")
        st.plotly_chart(fig_mkt, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-header">Threshold Sensitivity</div>', unsafe_allow_html=True)
        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(
            x=df_thresh["Threshold"],
            y=df_thresh["Future flagged percentage"],
            mode="lines+markers+text",
            line=dict(color=CYAN, width=2),
            marker=dict(size=10, color=WARN, line=dict(color=PLOT_BG, width=2)),
            text=[
                f"p{int(p)}<br>{v:.2f}%"
                for p, v in zip(
                    df_thresh["Training percentile"],
                    df_thresh["Future flagged percentage"],
                )
            ],
            textposition="top center",
            textfont=dict(color=FONT_CLR, size=10),
        ))
        fig_thr.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_color=WARN,
            line_width=1.5,
            annotation_text=f"Active ({THRESHOLD:.4f})",
            annotation_font_color=WARN,
        )
        fig_thr.update_layout(
            **base_layout(height=350),
            xaxis_title="Threshold Value",
            yaxis_title="% of Future Bookings Flagged",
        )
        st.plotly_chart(fig_thr, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Score by risk | Weekly flagged trend ──
    # ── Row 3: Score by risk | Weekly flagged trend ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Anomaly Score by Risk Level</div>', unsafe_allow_html=True)
        cap99 = df["anomaly_score"].quantile(0.99)
        df_bp = df[df["anomaly_score"] <= cap99].copy()
        fig_box = px.box(
            df_bp,
            x="risk_level",
            y="anomaly_score",
            color="risk_level",
            color_discrete_map={"Typical": TEAL, "Review": DANGER},
            points="outliers",
        )
        fig_box.add_hline(
            y=THRESHOLD, line_dash="dash", line_color=WARN,
            annotation_text=f"Threshold {THRESHOLD:.3f}",
            annotation_font_color=WARN,
        )
        fig_box.update_layout(
            **base_layout(height=320, showlegend=False),
            xaxis_title="Risk Level",
            yaxis_title="Anomaly Score",
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    with col_d:
        st.markdown('<div class="section-header">Weekly Flagged Booking Trend</div>', unsafe_allow_html=True)
        df_flagged = df[df["is_anomaly"] == 1].copy()
        df_flagged["week"] = df_flagged["date_time"].dt.to_period("W").dt.start_time
        weekly_flag = df_flagged.groupby("week").size().reset_index(name="flagged_count")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=weekly_flag["week"],
            y=weekly_flag["flagged_count"],
            mode="lines+markers",
            line=dict(color=DANGER, width=2),
            fill="tozeroy",
            fillcolor="rgba(255,79,114,0.12)",
            marker=dict(size=6, color=DANGER),
            name="Flagged / week",
        ))
        fig_trend.add_hline(
            y=weekly_flag["flagged_count"].mean(),
            line_dash="dot",
            line_color=WARN,
            annotation_text="Weekly avg",
            annotation_font_color=WARN,
        )
        fig_trend.update_layout(
            **base_layout(height=320, showlegend=False),
            xaxis_title="Week",
            yaxis_title="Flagged Bookings",
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Market cancellation rate vs anomaly score ──
    # Kept as the final analytical relationship chart. The low-value
    # flagged-user distribution chart has been removed to keep this page focused.
    st.markdown(
        '<div class="section-header">Market Cancellation Rate vs Anomaly Score</div>',
        unsafe_allow_html=True,
    )
    df_mkt_scat = df[["past_market_cancellation_rate", "anomaly_score", "is_anomaly"]].copy()
    df_mkt_scat = df_mkt_scat[
        df_mkt_scat["anomaly_score"] <= df_mkt_scat["anomaly_score"].quantile(0.98)
    ]
    sample_idx = df_mkt_scat.sample(min(800, len(df_mkt_scat)), random_state=7).index
    df_samp = df_mkt_scat.loc[sample_idx].copy()
    df_samp["Status"] = df_samp["is_anomaly"].map({0: "Normal", 1: "Flagged"})

    fig_mkt_s = px.scatter(
        df_samp,
        x="past_market_cancellation_rate",
        y="anomaly_score",
        color="Status",
        color_discrete_map={"Normal": TEAL, "Flagged": DANGER},
        opacity=0.65,
        labels={
            "past_market_cancellation_rate": "Market Cancellation Rate",
            "anomaly_score": "Anomaly Score",
        },
    )
    fig_mkt_s.add_hline(
        y=THRESHOLD, line_dash="dash", line_color=WARN,
        annotation_text=f"Threshold {THRESHOLD:.3f}",
        annotation_font_color=WARN,
    )
    fig_mkt_s.update_layout(**base_layout(height=390))
    st.plotly_chart(fig_mkt_s, use_container_width=True, config={"displayModeBar": False})

    # ── Pipeline Run Summary ──
    st.markdown('<div class="section-header">Pipeline Run Summary</div>', unsafe_allow_html=True)
    s_data = {
        "Metric": [
            "Source Travel Rows",
            "Source Booking Rows",
            "Synthetic Labels",
            "Chronological Cutoff",
            "Training Rows",
            "Future Test Rows",
            "Threshold (95th pct)",
            "Future Flagged Count",
            "Future Flagged %",
        ],
        "Value": [
            f"{summary['source_travel_rows']:,}",
            f"{summary['source_booking_rows']:,}",
            f"{summary['synthetic_label_count']:,}",
            summary["chronological_cutoff"][:19],
            f"{summary['training_rows']:,}",
            f"{summary['future_test_rows']:,}",
            f"{summary['threshold']:.6f}",
            f"{summary['future_flagged_count']:,}",
            f"{summary['future_flagged_percentage']:.2f}%",
        ],
    }
    df_sum = pd.DataFrame(s_data)
    st.dataframe(
        style_table(df_sum),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        f'<div class="disclaimer">📌 {summary["note"]}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# PAGE 5 – FRAUD INVESTIGATION
# Covers M4.1 (Model Evaluation) and M4.2 (Fraud-Ring Analysis)
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Fraud Investigation":

    st.markdown(
        '<h2 style="color:#00d4ff;margin-bottom:4px;">Fraud Investigation</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="color:#8ba0bb;font-size:0.82rem;">'
        'Evaluation of detection behaviour, booking pattern analysis, and model limitations. '
        'All findings are based on anomaly scores and <b>rule-based synthetic labels</b> — '
        'no verified fraud ground-truth is available.</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Load data ──────────────────────────────────────────────
    df       = load_anomaly_results()
    df_thr   = load_threshold_comparison()
    df_usr   = load_flagged_user()
    df_mkt   = load_flagged_market()

    flagged  = df[df["is_anomaly"] == 1].copy()
    normal   = df[df["is_anomaly"] == 0].copy()

    SYNTH_COL      = "synthetic_fraud_label"
    SYNTH_REASON   = "synthetic_label_reason"
    HAS_SYNTH      = SYNTH_COL in df.columns

    # ══════════════════════════════════════════════════════════
    # SECTION 1 – Normal vs Anomalous Booking Comparison (M4.2.1)
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Normal vs Anomalous Booking Comparison</div>',
        unsafe_allow_html=True,
    )

    # Build comparison metrics using only columns that exist
    compare_specs = [
        ("lead_time_days",               "Lead Time (days)",              "mean"),
        ("stay_duration_days",           "Stay Duration (days)",          "mean"),
        ("amount",                       "Booking Amount",                "mean"),
        ("num_guests",                   "Number of Guests",              "mean"),
        ("num_rooms",                    "Number of Rooms",               "mean"),
        ("past_user_cancellation_rate",  "User Cancellation Rate",        "mean"),
        ("past_market_cancellation_rate","Market Cancellation Rate",      "mean"),
        ("past_user_bookings",           "Past User Bookings",            "mean"),
        ("past_market_bookings",         "Past Market Bookings",          "mean"),
        ("anomaly_score",                "Anomaly Score (MSE)",           "mean"),
    ]

    rows = []
    for col, label, agg in compare_specs:
        if col not in df.columns:
            continue
        n_val = normal[col].mean()   if agg == "mean" else normal[col].median()
        f_val = flagged[col].mean()  if agg == "mean" else flagged[col].median()
        diff  = f_val - n_val
        pct   = (diff / n_val * 100) if n_val != 0 else float("nan")
        rows.append({
            "Feature":               label,
            "Normal (mean)":         round(n_val, 4),
            "Flagged (mean)":        round(f_val, 4),
            "Difference":            round(diff, 4),
            "% Change vs Normal":    f"{pct:+.1f}%" if not (pct != pct) else "—",
        })

    df_comp = pd.DataFrame(rows)

    # Colour-code the % Change column
    def style_pct(v):
        if isinstance(v, str) and v.startswith("+"):
            return "color:#ff4f72;font-weight:600"
        if isinstance(v, str) and v.startswith("-"):
            return "color:#28e0a4;font-weight:600"
        return ""

    st.dataframe(
        style_table(df_comp).applymap(style_pct, subset=["% Change vs Normal"]),
        use_container_width=True,
        hide_index=True,
    )

    comparison_lookup = {row["Feature"]: row for row in rows}
    lead_row = comparison_lookup.get("Lead Time (days)", {})
    amount_row = comparison_lookup.get("Booking Amount", {})
    guests_row = comparison_lookup.get("Number of Guests", {})
    rooms_row = comparison_lookup.get("Number of Rooms", {})
    st.markdown(
        f"""<div class="disclaimer">
        <b>Key observations</b><br>
        • Average lead time: <b>{lead_row.get('Normal (mean)', float('nan')):,.1f} days</b> for normal bookings
          vs <b>{lead_row.get('Flagged (mean)', float('nan')):,.1f} days</b> for flagged bookings.<br>
        • Average booking amount: <b>{amount_row.get('Normal (mean)', float('nan')):,.0f}</b> for normal
          vs <b>{amount_row.get('Flagged (mean)', float('nan')):,.0f}</b> for flagged bookings.<br>
        • Average guests and rooms: <b>{guests_row.get('Normal (mean)', float('nan')):,.2f} / {rooms_row.get('Normal (mean)', float('nan')):,.2f}</b>
          for normal vs <b>{guests_row.get('Flagged (mean)', float('nan')):,.2f} / {rooms_row.get('Flagged (mean)', float('nan')):,.2f}</b>
          for flagged bookings.<br>
        • These are population-level averages; individual flagged bookings may look similar to normal ones.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Side-by-side bar chart – normalised (z-score) so all features fit one scale
    comp_features = [r["Feature"] for r in rows if r["Feature"] != "Anomaly Score (MSE)"]
    normal_means  = [r["Normal (mean)"]  for r in rows if r["Feature"] != "Anomaly Score (MSE)"]
    flagged_means = [r["Flagged (mean)"] for r in rows if r["Feature"] != "Anomaly Score (MSE)"]

    # z-score relative to overall column std to make bars comparable
    norm_means_arr  = np.array(normal_means,  dtype=float)
    flag_means_arr  = np.array(flagged_means, dtype=float)

    stds = []
    feat_col_map = {r["Feature"]: spec[0] for r, spec in zip(rows, compare_specs)
                    if r["Feature"] != "Anomaly Score (MSE)"}
    for lbl in comp_features:
        col_name = feat_col_map.get(lbl)
        stds.append(df[col_name].std() if col_name and col_name in df.columns else 1.0)
    stds_arr = np.where(np.array(stds) == 0, 1.0, stds)

    z_normal  = (norm_means_arr  - norm_means_arr)  / stds_arr   # always 0 (baseline)
    z_flagged = (flag_means_arr  - norm_means_arr)  / stds_arr

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Normal (baseline = 0)",
        x=comp_features,
        y=z_normal,
        marker_color=TEAL,
        opacity=0.80,
    ))
    fig_comp.add_trace(go.Bar(
        name="Flagged (deviation from normal)",
        x=comp_features,
        y=z_flagged,
        marker_color=DANGER,
        opacity=0.85,
    ))
    fig_comp.add_hline(y=0, line_color=TEAL, line_dash="dot", line_width=1)
    fig_comp.update_layout(
        **base_layout(height=360, barmode="group"),
        xaxis_title="Feature",
        yaxis_title="Deviation from Normal Mean (std units)",
        xaxis_tickangle=-30,
        title=dict(text="Flagged vs Normal — Normalised Feature Deviation", font_color=FONT_CLR, font_size=13),
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════
    # SECTION 2 – Coordinated Booking Pattern Analysis (M4.2.2)
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Coordinated Booking Pattern Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="disclaimer">
        ⚠️ <b>Important caveat:</b> Shared markets, similar booking dates, or similar feature values
        do not establish coordinated behaviour. The analysis below identifies <i>correlations in the data</i>
        only. No user is labelled a fraudster, and no fraud ring is claimed.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Markets with multiple distinct flagged users
    mkt_summary = (
        flagged
        .groupby("hotel_market", as_index=False)
        .agg(
            flagged_bookings=("booking_id", "count"),
            distinct_users=("user_id", "nunique"),
            mean_score=("anomaly_score", "mean"),
            mean_lead_time=("lead_time_days", "mean"),
            mean_amount=("amount", "mean"),
        )
    )
    multi_mkt = (
        mkt_summary[mkt_summary["distinct_users"] > 1]
        .sort_values("distinct_users", ascending=False)
        .reset_index(drop=True)
    )
    multi_mkt["mean_score"]     = multi_mkt["mean_score"].round(3)
    multi_mkt["mean_lead_time"] = multi_mkt["mean_lead_time"].round(1)
    multi_mkt["mean_amount"]    = multi_mkt["mean_amount"].round(0).astype(int)

    st.markdown(
        f'<div class="section-header" style="font-size:0.95rem;">'
        f'Travel Markets with Multiple Flagged Users ({len(multi_mkt)} markets)</div>',
        unsafe_allow_html=True,
    )

    if multi_mkt.empty:
        st.info("No travel market contains more than one distinct flagged user. "
                "No multi-user market pattern is detectable from available data.")
    else:
        multi_mkt.columns = [
            "Market ID", "Flagged Bookings", "Distinct Users",
            "Mean Anomaly Score", "Mean Lead Time (days)", "Mean Booking Amount",
        ]
        st.dataframe(
            style_table(multi_mkt).applymap(
                lambda v: "color:#a85d00;font-weight:600" if isinstance(v, (int, np.integer)) and v >= 3 else "",
                subset=["Distinct Users"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Bubble chart – markets with multiple flagged users
        col_p, col_q = st.columns(2)
        with col_p:
            fig_mlt = px.scatter(
                multi_mkt,
                x="Mean Lead Time (days)",
                y="Mean Anomaly Score",
                size="Flagged Bookings",
                color="Distinct Users",
                color_continuous_scale=[[0, TEAL], [0.5, WARN], [1, DANGER]],
                hover_name="Market ID",
                labels={"Mean Lead Time (days)": "Mean Lead Time (days)",
                        "Mean Anomaly Score": "Mean Anomaly Score"},
                size_max=28,
            )
            fig_mlt.update_layout(
                **base_layout(height=320),
                title=dict(text="Multi-User Markets: Lead Time vs Anomaly Score",
                           font_color=FONT_CLR, font_size=12),
                coloraxis_colorbar=dict(title="Distinct Users", tickfont_color=FONT_CLR),
            )
            st.plotly_chart(fig_mlt, use_container_width=True, config={"displayModeBar": False})

        with col_q:
            # Show the individual flagged bookings in those markets
            shared_mkt_ids = multi_mkt["Market ID"].tolist()
            flagged_shared = flagged[flagged["hotel_market"].isin(shared_mkt_ids)][
                ["date_time", "user_id", "hotel_market", "lead_time_days",
                 "amount", "num_guests", "anomaly_score"]
            ].sort_values(["hotel_market", "date_time"]).reset_index(drop=True)
            flagged_shared["anomaly_score"] = flagged_shared["anomaly_score"].round(3)
            flagged_shared["amount"]        = flagged_shared["amount"].round(0).astype(int)

            st.markdown(
                '<div style="font-size:0.82rem;color:#8ba0bb;margin-bottom:6px;">'
                f'Individual flagged bookings in shared markets ({len(flagged_shared)} bookings)</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(style_table(flagged_shared), use_container_width=True, hide_index=True, height=300)

    # Repeated cancellation pattern
    st.markdown("")
    st.markdown(
        '<div style="font-size:0.95rem;font-weight:600;color:#00d4ff;border-bottom:1px solid #1e3250;'
        'padding-bottom:4px;margin:12px 0 10px 0;">Repeated Cancellation Behaviour</div>',
        unsafe_allow_html=True,
    )
    cancel_col = "is_cancelled" if "is_cancelled" in df.columns else None
    if cancel_col:
        cancel_flagged = flagged[cancel_col].sum()
        cancel_normal  = normal[cancel_col].sum()
        cancel_rate_f  = cancel_flagged / len(flagged) * 100 if len(flagged) else 0.0
        cancel_rate_n  = cancel_normal  / len(normal)  * 100 if len(normal) else 0.0
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Flagged Cancellations", f"{int(cancel_flagged)}",
                     f"{cancel_rate_f:.1f}% of flagged", "danger")
        with c2: kpi("Normal Cancellations",  f"{int(cancel_normal)}",
                     f"{cancel_rate_n:.1f}% of normal", "ok")
        with c3: kpi("Cancellation Ratio",
                     f"{cancel_rate_f/cancel_rate_n:.1f}×" if cancel_rate_n > 0 else "—",
                     "flagged vs normal rate", "warn")
    else:
        st.info("Cancellation column not found in the data.")

    # ══════════════════════════════════════════════════════════
    # SECTION 3 – False Positive Analysis (M4.1.2)
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">False Positive Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="disclaimer">
        The model flags bookings whose reconstruction error exceeds the 95th-percentile training threshold.
        <b>A flag does not confirm fraud.</b> Any flagged booking could be an unusual-but-legitimate
        transaction. Without verified fraud labels, false-positive <i>rates</i> cannot be precisely
        calculated — only estimates based on score proximity to the threshold can be provided.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Score band breakdown
    score_bins = [THRESHOLD, 1.5, 2.0, 3.0, 5.0, 10.0,
                  max(10.01, float(flagged["anomaly_score"].max()) + 0.01)]
    score_labels = [f"{THRESHOLD:.2f}–1.5 (near threshold)", "1.5–2.0", "2.0–3.0",
                    "3.0–5.0", "5.0–10.0", ">10.0 (high anomaly score)"]
    flagged["_score_band"] = pd.cut(
        flagged["anomaly_score"], bins=score_bins, labels=score_labels, right=True,
        include_lowest=False
    )
    band_counts = (
        flagged["_score_band"]
        .value_counts()
        .reindex(score_labels, fill_value=0)
        .reset_index()
    )
    band_counts.columns = ["Score Band", "Flagged Bookings"]
    band_counts["% of Flagged"] = (band_counts["Flagged Bookings"] / len(flagged) * 100).round(1)
    band_counts["Review Priority"] = [
        "Lowest – near-threshold review priority",
        "Low",
        "Medium",
        "High",
        "High",
        "Highest – strong anomaly signal",
    ]

    fa1, fa2 = st.columns([2, 1])
    with fa1:
        st.markdown(
            '<div style="font-size:0.9rem;color:#8ba0bb;margin-bottom:6px;">'
            'Flagged Bookings by Anomaly Score Band</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(style_table(band_counts), use_container_width=True, hide_index=True)

    with fa2:
        fig_band = px.bar(
            band_counts,
            x="Flagged Bookings",
            y="Score Band",
            orientation="h",
            color="Flagged Bookings",
            color_continuous_scale=[[0, TEAL], [0.5, WARN], [1, DANGER]],
        )
        fig_band.update_layout(
            **base_layout(height=290, showlegend=False, margin=dict(l=10, r=30, t=20, b=20)),
            xaxis_title="Count",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_band, use_container_width=True, config={"displayModeBar": False})

    # Flagged booking inspection table
    st.markdown(
        '<div style="font-size:0.9rem;color:#8ba0bb;margin:14px 0 6px 0;">'
        'Flagged Bookings for Manual Inspection (sorted by anomaly score)</div>',
        unsafe_allow_html=True,
    )
    inspect_cols = [c for c in [
        "date_time", "booking_id", "user_id", "hotel_market",
        "lead_time_days", "stay_duration_days", "amount", "num_guests", "num_rooms",
        "past_user_cancellation_rate", "past_market_cancellation_rate",
        "anomaly_score", "risk_level",
    ] if c in flagged.columns]

    df_inspect = flagged[inspect_cols].sort_values("anomaly_score", ascending=False).copy()
    df_inspect["anomaly_score"] = df_inspect["anomaly_score"].round(4)
    if "amount" in df_inspect.columns:
        df_inspect["amount"] = df_inspect["amount"].round(0).astype(int)
    st.dataframe(
        style_table(df_inspect).applymap(
            lambda v: "color:#c62845;font-weight:600" if v == "Review" else "",
            subset=[c for c in ["risk_level"] if c in df_inspect.columns],
        ),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    # Scatter: anomaly score with threshold + near-threshold callout
    st.markdown("")
    df_scatter = flagged.copy()
    df_scatter["Near Threshold"] = df_scatter["anomaly_score"].apply(
        lambda x: f"Near Threshold ({THRESHOLD:.2f}–1.5)" if x <= 1.5 else "Higher Anomaly Score (>1.5)"
    )
    fig_fp = px.scatter(
        df_scatter.sort_values("anomaly_score"),
        x=range(len(df_scatter)),
        y="anomaly_score",
        color="Near Threshold",
        color_discrete_map={
            f"Near Threshold ({THRESHOLD:.2f}–1.5)": WARN,
            "Higher Anomaly Score (≥1.5)": DANGER,
        },
        hover_data={"anomaly_score": True, "user_id": True, "hotel_market": True},
        labels={"x": "Flagged Booking Index (sorted by score)", "anomaly_score": "Anomaly Score"},
    )
    fig_fp.add_hline(
        y=THRESHOLD, line_dash="dash", line_color=TEAL, line_width=1.5,
        annotation_text=f"Detection threshold ({THRESHOLD:.4f})",
        annotation_font_color=TEAL,
    )
    fig_fp.update_layout(
        **base_layout(height=300),
        title=dict(
            text=f"All {len(flagged)} Flagged Bookings — Score vs Threshold (near-threshold = review priority)",
            font_color=FONT_CLR, font_size=12,
        ),
    )
    st.plotly_chart(fig_fp, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════
    # SECTION 4 – Detection Metrics & Threshold Sensitivity (M4.1.1 / M4.1.3)
    # ══════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-header">Detection Metrics & Threshold Sensitivity</div>',
        unsafe_allow_html=True,
    )

    # ── 4a: Detection counts ────────────────────────────────
    st.markdown(
        '<div style="font-size:0.9rem;color:#8ba0bb;margin-bottom:8px;">'
        '<b style="color:#00c8b4">Detection Counts at Active Threshold (95th percentile)</b></div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi("Total Screened",   f"{len(df):,}",     "future test set")
    with m2: kpi("Flagged",          f"{len(flagged):,}", f"{len(flagged)/len(df)*100:.2f}% of total", "danger")
    with m3: kpi("Not Flagged",      f"{len(normal):,}",  f"{len(normal)/len(df)*100:.2f}% of total",  "ok")
    with m4: kpi("Active Threshold", f"{THRESHOLD:.4f}",  "95th-pct training MSE", "warn")

    st.markdown("")

    # ── 4b: Synthetic-label metrics (clearly caveated) ─────
    if HAS_SYNTH:
        st.markdown(
            '<div style="font-size:0.9rem;color:#8ba0bb;margin:12px 0 6px 0;">'
            '<b style="color:#ffa040">Metrics Against Synthetic (Rule-Based) Labels — '
            'NOT verified fraud ground-truth</b></div>',
            unsafe_allow_html=True,
        )
        tp = int(((df["is_anomaly"] == 1) & (df[SYNTH_COL] == 1)).sum())
        fp = int(((df["is_anomaly"] == 1) & (df[SYNTH_COL] == 0)).sum())
        fn = int(((df["is_anomaly"] == 0) & (df[SYNTH_COL] == 1)).sum())
        tn = int(((df["is_anomaly"] == 0) & (df[SYNTH_COL] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Confusion table
        conf_df = pd.DataFrame(
            {
                "":                        ["Anomaly model: FLAGGED", "Anomaly model: NOT FLAGGED"],
                "Synthetic label: FRAUD":  [tp, fn],
                "Synthetic label: NORMAL": [fp, tn],
            }
        )

        sa1, sa2 = st.columns([1, 1])
        with sa1:
            st.dataframe(style_table(conf_df), use_container_width=True, hide_index=True)

        with sa2:
            sm1, sm2, sm3 = st.columns(3)
            with sm1: kpi("Synth. Precision", f"{precision:.3f}", "TP/(TP+FP) vs synth.", "warn")
            with sm2: kpi("Synth. Recall",    f"{recall:.3f}",    "TP/(TP+FN) vs synth.", "warn")
            with sm3: kpi("Synth. F1",        f"{f1:.3f}",        "harmonic mean", "warn")

        st.markdown(
            """<div class="disclaimer">
            These metrics compare the autoencoder's flags against <b>rule-based synthetic labels</b>
            generated during data supplementation — not against real, verified fraud outcomes.
            Low precision and recall against synthetic labels are expected: the autoencoder detects
            reconstruction anomalies, while the synthetic rules flag a different (overlapping but
            not identical) set of patterns. These numbers should <b>not</b> be interpreted as the
            model's real-world fraud-detection performance.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")

    # ── 4c: Threshold sensitivity (reused from Analytics page) ─
    st.markdown(
        '<div style="font-size:0.9rem;color:#8ba0bb;margin:12px 0 6px 0;">'
        '<b style="color:#00c8b4">Threshold Sensitivity — Impact of Threshold Choice</b></div>',
        unsafe_allow_html=True,
    )
    df_thr_disp = df_thr.copy()
    df_thr_disp.columns = [c.strip() for c in df_thr_disp.columns]

    thr_col  = "Threshold"
    pct_col  = "Future flagged percentage"
    flag_col = "Future flagged bookings"
    pct_col2 = "Training percentile"

    thr1, thr2 = st.columns([2, 1])
    with thr1:
        fig_thr2 = go.Figure()
        fig_thr2.add_trace(go.Scatter(
            x=df_thr_disp[thr_col],
            y=df_thr_disp[pct_col],
            mode="lines+markers+text",
            line=dict(color=CYAN, width=2),
            marker=dict(size=10, color=WARN, line=dict(color=PLOT_BG, width=2)),
            text=[f"p{int(p)}: {v:.2f}%" for p, v in
                  zip(df_thr_disp[pct_col2], df_thr_disp[pct_col])],
            textposition="top center",
            textfont=dict(color=FONT_CLR, size=10),
        ))
        fig_thr2.add_vline(
            x=THRESHOLD, line_dash="dash", line_color=WARN, line_width=1.5,
            annotation_text=f"Active ({THRESHOLD:.4f})",
            annotation_font_color=WARN,
        )
        fig_thr2.update_layout(
            **base_layout(height=300),
            xaxis_title="Threshold Value",
            yaxis_title="% of Bookings Flagged",
        )
        st.plotly_chart(fig_thr2, use_container_width=True, config={"displayModeBar": False})

    with thr2:
        thr_table = df_thr_disp[[pct_col2, thr_col, flag_col, pct_col]].copy()
        thr_table.columns = ["Percentile", "Threshold", "Flagged", "% Flagged"]
        thr_table["Threshold"] = thr_table["Threshold"].round(4)
        thr_table["% Flagged"] = thr_table["% Flagged"].round(2)
        st.dataframe(style_table(thr_table), use_container_width=True, hide_index=True)
        st.markdown(
            '<div style="font-size:0.76rem;color:#8ba0bb;margin-top:6px;">'
            'Raising the threshold lowers the flag rate but increases the risk of missed anomalies.</div>',
            unsafe_allow_html=True,
        )

   