"""
╔══════════════════════════════════════════════════════════════╗
║  RIDE DEMAND INTELLIGENCE — Premium Analytics Dashboard     ║
║  Engineered with Streamlit · Plotly · Folium                ║
║  Glassmorphism + Animated Gradients + Pro Data Viz          ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import joblib
import os
import sys
import warnings
from datetime import datetime, timedelta
import time
import math

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import MultiHorizonForecaster, SurgePricingModel
from src.driver_allocation import DriverAllocator
from src.utils import CITY_LAT_MIN, CITY_LAT_MAX, CITY_LON_MIN, CITY_LON_MAX


# ═══════════════════════════════════════════════════════════
#  AUTO-BOOTSTRAP (for Streamlit Cloud — data/ & models/ are gitignored)
# ═══════════════════════════════════════════════════════════
def _ensure_pipeline():
    """Run the full pipeline if data or models are missing (first Cloud boot)."""
    data_exists = os.path.exists("data/processed_demand.csv")
    models_exist = os.path.exists("models/best_model.pkl") or os.path.exists("models/lgbm_15m.pkl")
    if data_exists and models_exist:
        return
    from src.data_generation import generate_synthetic_data
    from src.data_processing import process_data
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("data/raw_rides.csv"):
        generate_synthetic_data(num_rides=20000)
    if not os.path.exists("data/processed_demand.csv"):
        process_data(interval='15min')
    if not models_exist:
        df = pd.read_csv("data/processed_demand.csv")
        forecaster = MultiHorizonForecaster()
        forecaster.train(df)
        forecaster.save("models")

_ensure_pipeline()

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RidePulse · Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
#  PREMIUM CSS — Glassmorphism, Animated Gradients, Pro UI
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #050510;
        --bg-secondary: #0a0a1a;
        --bg-card: rgba(15, 15, 35, 0.65);
        --bg-glass: rgba(255,255,255,0.03);
        --border-glass: rgba(255,255,255,0.06);
        --accent: #7c6aff;
        --accent-glow: rgba(124,106,255,0.25);
        --cyan: #00e5ff;
        --cyan-glow: rgba(0,229,255,0.2);
        --rose: #ff5c8a;
        --rose-glow: rgba(255,92,138,0.2);
        --amber: #ffb84d;
        --emerald: #00e68a;
        --text-primary: #f0f0f5;
        --text-muted: rgba(240,240,245,0.5);
        --text-dim: rgba(240,240,245,0.3);
        --radius: 16px;
        --radius-sm: 10px;
        --shadow-lg: 0 20px 60px rgba(0,0,0,0.5);
        --shadow-glow: 0 0 40px var(--accent-glow);
    }

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* ─── Hero Header ─── */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0f0a2e 0%, #1a103d 30%, #0d1b3e 60%, #0a0a1a 100%);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.8rem;
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-lg);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: float 8s ease-in-out infinite;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, var(--cyan-glow) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        animation: float 6s ease-in-out infinite reverse;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #fff 0%, #c4b5fd 50%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .hero-sub {
        position: relative;
        z-index: 1;
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.3px;
    }
    .hero-badge {
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-top: 1rem;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        background: rgba(124,106,255,0.12);
        color: var(--accent);
        border: 1px solid rgba(124,106,255,0.2);
    }

    /* ─── Glass Cards ─── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        padding: 1.6rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.5s ease-out;
    }
    .glass-card:hover {
        border-color: rgba(124,106,255,0.15);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3), var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ─── KPI Metric Cards ─── */
    .kpi-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        padding: 1.4rem 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        animation: slideUp 0.4s ease-out backwards;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius) var(--radius) 0 0;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.3);
    }
    .kpi-card.purple::before { background: linear-gradient(90deg, var(--accent), #a78bfa); }
    .kpi-card.cyan::before { background: linear-gradient(90deg, var(--cyan), #67e8f9); }
    .kpi-card.rose::before { background: linear-gradient(90deg, var(--rose), #fda4af); }
    .kpi-card.amber::before { background: linear-gradient(90deg, var(--amber), #fcd34d); }
    .kpi-card.emerald::before { background: linear-gradient(90deg, var(--emerald), #6ee7b7); }
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0.2rem 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 600;
        color: var(--text-muted);
    }
    .kpi-trend {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .trend-up { background: rgba(0,230,138,0.12); color: var(--emerald); }
    .trend-down { background: rgba(255,92,138,0.12); color: var(--rose); }
    .trend-neutral { background: rgba(255,184,77,0.12); color: var(--amber); }

    /* ─── Section Separators ─── */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 2rem 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border-glass);
    }
    .section-title h3 {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.3px;
    }
    .section-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .icon-purple { background: rgba(124,106,255,0.15); }
    .icon-cyan { background: rgba(0,229,255,0.15); }
    .icon-rose { background: rgba(255,92,138,0.15); }
    .icon-amber { background: rgba(255,184,77,0.15); }

    /* ─── Data Table ─── */
    .table-container {
        background: var(--bg-card);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius);
        overflow: hidden;
    }

    /* ─── Status / Tags ─── */
    .tag {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .tag-live {
        background: rgba(0,229,255,0.1);
        color: var(--cyan);
        border: 1px solid rgba(0,229,255,0.2);
        animation: pulse 2s ease-in-out infinite;
    }
    .tag-model {
        background: rgba(124,106,255,0.1);
        color: var(--accent);
        border: 1px solid rgba(124,106,255,0.2);
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08081a 0%, #0d0d24 50%, #0a0a18 100%);
        border-right: 1px solid var(--border-glass);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .sidebar-brand h2 {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent), var(--cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-brand p {
        color: var(--text-dim);
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0.2rem 0 0 0;
    }
    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
        margin: 1rem 0;
    }

    /* ─── Misc ─── */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-right: 0.5rem;
    }
    .stat-pill strong { color: var(--text-primary); font-weight: 700; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border-glass); }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Plotly overrides */
    .js-plotly-plot { border-radius: var(--radius-sm); overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def load_data():
    processed_df, raw_df = None, None
    if os.path.exists("data/processed_demand.csv"):
        processed_df = pd.read_csv("data/processed_demand.csv")
        processed_df['time_bin'] = pd.to_datetime(processed_df['time_bin'])
    if os.path.exists("data/raw_rides.csv"):
        raw_df = pd.read_csv("data/raw_rides.csv")
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    return processed_df, raw_df


@st.cache_resource
def load_models():
    forecaster = MultiHorizonForecaster()
    try:
        forecaster.load("models")
    except Exception:
        forecaster = None
    kmeans = joblib.load("models/kmeans_model.pkl") if os.path.exists("models/kmeans_model.pkl") else None
    return forecaster, kmeans, DriverAllocator(), SurgePricingModel()


def kpi(icon, value, label, trend_text="", trend_dir="up", color="purple", delay=0):
    trend_cls = f"trend-{trend_dir}"
    trend_html = f'<div class="kpi-trend {trend_cls}">{trend_text}</div>' if trend_text else ""
    return f"""
    <div class="kpi-card {color}" style="animation-delay:{delay}s">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {trend_html}
    </div>"""


def section(icon, title, icon_color="purple"):
    return f"""
    <div class="section-title">
        <div class="section-icon icon-{icon_color}">{icon}</div>
        <h3>{title}</h3>
    </div>"""


_BASE_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, -apple-system, sans-serif", color="rgba(240,240,245,0.7)"),
)
_GRID = dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)')
_DEFAULT_MARGIN = dict(l=50, r=20, t=30, b=40)


def chart_layout(**overrides):
    """Build a merged layout dict. Overrides win over defaults."""
    base = dict(_BASE_LAYOUT)
    base['margin'] = dict(_DEFAULT_MARGIN)
    base['xaxis'] = dict(_GRID)
    base['yaxis'] = dict(_GRID)
    for k, v in overrides.items():
        if k in ('xaxis', 'yaxis') and isinstance(v, dict):
            base[k] = {**base.get(k, {}), **v}
        elif k == 'margin' and isinstance(v, dict):
            base['margin'] = {**base['margin'], **v}
        else:
            base[k] = v
    return base

COLOR_SEQ = ['#7c6aff', '#00e5ff', '#ff5c8a', '#ffb84d', '#00e68a', '#a78bfa',
             '#67e8f9', '#fda4af', '#fcd34d', '#6ee7b7']


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>⚡ RidePulse</h2>
        <p>Intelligence Platform</p>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Command Center",
        "🗺️ Geospatial Intel",
        "⚡ Surge Simulator",
        "🧠 ML Observatory",
        "📈 Time Series Lab",
        "📋 Data Explorer",
    ], index=0, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("##### ⚙️ Simulation Parameters")
    total_drivers = st.slider("Fleet Size", 50, 500, 150, step=10)
    avg_fare = st.slider("Avg Base Fare ($)", 5.0, 50.0, 15.0, step=0.5)
    sim_hour = st.slider("Hour of Day", 0, 23, 17)
    sim_day = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=4)
    day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0;">
        <span class="tag tag-live">● LIVE</span>
        <span class="tag tag-model" style="margin-left:4px;">v2.0</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════
processed_df, raw_df = load_data()
forecaster, kmeans, allocator, surge_model = load_models()
data_ok = processed_df is not None and not processed_df.empty
models_ok = forecaster is not None and len(forecaster.models) > 0
CENTER_LAT = (CITY_LAT_MIN + CITY_LAT_MAX) / 2
CENTER_LON = (CITY_LON_MIN + CITY_LON_MAX) / 2


def run_sim(fleet, hour, dow):
    if kmeans is None or forecaster is None:
        return None, None
    centroids = kmeans.cluster_centers_
    rows = []
    for i, (lat, lon) in enumerate(centroids):
        rows.append({
            'zone_id': str(i), 'zone_lat': lat, 'zone_lon': lon,
            'hour': hour, 'minute': 0, 'day_of_week': dow,
            'is_weekend': 1 if dow >= 5 else 0, 'month': 2,
            'hour_sin': np.sin(2*np.pi*hour/24), 'hour_cos': np.cos(2*np.pi*hour/24),
            'day_sin': np.sin(2*np.pi*dow/7), 'day_cos': np.cos(2*np.pi*dow/7),
            'lag_1': np.random.poisson(12), 'lag_4': np.random.poisson(12),
            'lag_96': np.random.poisson(12), 'rolling_mean_4': np.random.poisson(12),
        })
    df = pd.DataFrame(rows)
    try:
        p = forecaster.predict(df)
        df['predicted_demand'] = p['15m']
        df['pred_30m'] = p.get('30m', p['15m'])
        df['pred_60m'] = p.get('60m', p['15m'])
    except Exception:
        df['predicted_demand'] = np.random.poisson(10, len(df))
        df['pred_30m'] = df['predicted_demand'] * 1.1
        df['pred_60m'] = df['predicted_demand'] * 1.3
    a = allocator.optimize_allocation(df.copy(), fleet)
    m = allocator.simulate_revenue(a, avg_fare=avg_fare)
    return a, m


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 1: COMMAND CENTER                                 │
# └─────────────────────────────────────────────────────────┘
if page == "📊 Command Center":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Command Center</div>
        <div class="hero-sub">Real-time demand forecasting, surge pricing & fleet optimization at a glance</div>
        <div class="hero-badge">Multi-Horizon AI · 20 Zones · Live Simulation</div>
    </div>""", unsafe_allow_html=True)

    if not data_ok:
        st.error("No data found. Run `python main.py` to generate data & train models.")
        st.stop()

    alloc_df, rev = run_sim(total_drivers, sim_hour, day_map[sim_day])

    # ─── KPIs ───
    if alloc_df is not None and rev:
        td = alloc_df['predicted_demand'].sum()
        shortage = int((alloc_df['gap'] > 5).sum()) if 'gap' in alloc_df.columns else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(kpi("💰", f"${rev['Total_Revenue']:,.0f}", "Projected Revenue", "15-min window", "up", "purple", 0), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("📊", f"{rev['Service_Level']:.1f}%", "Service Level", "fulfillment rate", "up", "cyan", 0.1), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("⚡", f"{rev['Avg_Surge']:.2f}x", "Avg Surge", "city-wide", "neutral", "amber", 0.2), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("📈", f"{td:.0f}", "Total Demand", f"{len(alloc_df)} zones", "up", "emerald", 0.3), unsafe_allow_html=True)
        with c5:
            st.markdown(kpi("🔴", f"{shortage}", "Shortage Zones", "need drivers", "down" if shortage > 3 else "up", "rose", 0.4), unsafe_allow_html=True)

        st.markdown("")

        # ─── Demand vs Supply + Surge ───
        left, right = st.columns([3, 2])
        with left:
            st.markdown(section("📈", "Demand vs Allocated Supply", "purple"), unsafe_allow_html=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=alloc_df['zone_id'].astype(str), y=alloc_df['predicted_demand'],
                name='Demand', marker=dict(color=alloc_df['predicted_demand'],
                colorscale=[[0,'#1a103d'],[0.5,'#7c6aff'],[1,'#c4b5fd']], line=dict(width=0)),
                opacity=0.85,
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=alloc_df['zone_id'].astype(str), y=alloc_df['allocated_drivers'],
                name='Drivers', mode='lines+markers',
                line=dict(color='#00e5ff', width=2.5, dash='dot'),
                marker=dict(size=7, symbol='diamond', line=dict(width=1, color='#00e5ff')),
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=alloc_df['zone_id'].astype(str), y=alloc_df['surge_multiplier'],
                name='Surge', mode='lines', fill='tozeroy',
                line=dict(color='#ff5c8a', width=1.5),
                fillcolor='rgba(255,92,138,0.08)',
            ), secondary_y=True)
            fig.update_layout(**chart_layout(height=400,
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
                xaxis_title="Zone ID", yaxis_title="Count"))
            fig.update_yaxes(title_text="Surge", secondary_y=True, range=[0.8, 3.5])
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown(section("🔥", "Surge Heatbar", "rose"), unsafe_allow_html=True)
            sorted_a = alloc_df.sort_values('surge_multiplier', ascending=True)
            colors = ['#00e68a' if s<=1.1 else '#ffb84d' if s<=1.5 else '#ff5c8a' for s in sorted_a['surge_multiplier']]
            fig_s = go.Figure(go.Bar(
                y=sorted_a['zone_id'].astype(str), x=sorted_a['surge_multiplier'],
                orientation='h', marker_color=colors,
                text=sorted_a['surge_multiplier'].round(2).astype(str)+'x',
                textposition='outside', textfont=dict(size=10, family='JetBrains Mono'),
            ))
            fig_s.add_vline(x=1.0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
            fig_s.update_layout(**chart_layout(height=400, showlegend=False,
                xaxis_title="Multiplier", yaxis_title="Zone"))
            st.plotly_chart(fig_s, width="stretch")

        # ─── Allocation Table ───
        st.markdown(section("🎯", "Zone Allocation Matrix", "cyan"), unsafe_allow_html=True)
        cols = ['zone_id','predicted_demand','allocated_drivers','gap','surge_multiplier','action']
        av = [c for c in cols if c in alloc_df.columns]
        tbl = alloc_df[av].copy()
        tbl.columns = ['Zone','Demand','Drivers','Gap','Surge','Action'][:len(av)]
        st.dataframe(tbl, height=320, use_container_width=True)

        # ─── Hourly Pattern ───
        st.markdown(section("🕐", "24-Hour Demand Rhythm", "amber"), unsafe_allow_html=True)
        hourly = processed_df.groupby('hour')['demand'].mean().reset_index()
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['demand'],
            mode='lines', fill='tozeroy',
            line=dict(color='#7c6aff', width=3, shape='spline'),
            fillcolor='rgba(124,106,255,0.1)',
        ))
        fig_h.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['demand'],
            mode='markers', marker=dict(size=6, color='#c4b5fd',
            line=dict(width=2, color='#7c6aff')), showlegend=False,
        ))
        peak_idx = hourly['demand'].idxmax()
        fig_h.add_annotation(x=hourly.loc[peak_idx,'hour'], y=hourly.loc[peak_idx,'demand'],
            text=f"Peak: {hourly.loc[peak_idx,'demand']:.1f}", showarrow=True,
            arrowhead=2, arrowcolor='#ff5c8a', font=dict(color='#ff5c8a', size=11))
        fig_h.update_layout(**chart_layout(height=280, xaxis=dict(dtick=2, title="Hour of Day"),
            yaxis_title="Avg Demand", showlegend=False))
        st.plotly_chart(fig_h, width="stretch")


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 2: GEOSPATIAL INTEL                               │
# └─────────────────────────────────────────────────────────┘
elif page == "🗺️ Geospatial Intel":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Geospatial Intelligence</div>
        <div class="hero-sub">High-fidelity spatial analysis of ride demand and zone clustering</div>
        <div class="hero-badge">K-Means Clustering · Heatmap Density · Centroid Analysis</div>
    </div>""", unsafe_allow_html=True)

    if raw_df is None:
        st.error("No raw data. Run `python main.py` first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔥 Demand Heatmap", "📍 Zone Clusters", "📊 Scatter Density"])

    with tab1:
        st.markdown(section("🔥", "Pickup Density Heatmap", "rose"), unsafe_allow_html=True)
        n = st.slider("Points to render", 300, min(5000, len(raw_df)), min(2000, len(raw_df)), 100)
        s = raw_df.sample(n, random_state=42)
        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13, tiles='CartoDB dark_matter')
        HeatMap(s[['pickup_lat','pickup_long']].values.tolist(), radius=13, blur=16,
                gradient={'0.2':'#0d0887','0.4':'#7201a8','0.6':'#bd3786','0.8':'#ed7953','1.0':'#fdca26'}).add_to(m)
        st_folium(m, height=560, use_container_width=True, key="demand_heatmap_map")

    with tab2:
        st.markdown(section("📍", "K-Means Cluster Centroids", "cyan"), unsafe_allow_html=True)
        if kmeans is not None:
            centroids = kmeans.cluster_centers_
            m2 = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13, tiles='CartoDB dark_matter')
            for i, (lat, lon) in enumerate(centroids):
                c = COLOR_SEQ[i % len(COLOR_SEQ)]
                folium.CircleMarker(location=[lat,lon], radius=16,
                    popup=f"<b style='color:{c}'>Zone {i}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}",
                    color=c, fill=True, fill_color=c, fill_opacity=0.65, weight=2).add_to(m2)
                folium.Marker(location=[lat,lon],
                    icon=folium.DivIcon(html=f'<div style="font-size:11px;font-weight:800;color:#fff;text-align:center;text-shadow:0 1px 4px rgba(0,0,0,0.8);">{i}</div>')).add_to(m2)
            st_folium(m2, height=560, use_container_width=True, key="zone_clusters_map")
        else:
            st.info("KMeans model not found.")

    with tab3:
        st.markdown(section("📊", "Pickup Scatter Plot", "purple"), unsafe_allow_html=True)
        s = raw_df.sample(min(3000, len(raw_df)), random_state=42)
        fig_sc = px.scatter_mapbox(s, lat='pickup_lat', lon='pickup_long',
            color_discrete_sequence=['#7c6aff'], opacity=0.4,
            mapbox_style='carto-darkmatter', zoom=12,
            center=dict(lat=CENTER_LAT, lon=CENTER_LON))
        fig_sc.update_layout(**chart_layout(height=550, margin=dict(l=0,r=0,t=0,b=0)))
        st.plotly_chart(fig_sc, width="stretch")

    # Distribution row
    st.markdown(section("📊", "Coordinate Distributions", "amber"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_lat = px.histogram(raw_df.sample(min(3000,len(raw_df))), x='pickup_lat', nbins=60,
            color_discrete_sequence=['#7c6aff'], template='plotly_dark')
        fig_lat.update_layout(**chart_layout(height=260, title="Latitude", title_font_size=13))
        st.plotly_chart(fig_lat, width="stretch")
    with c2:
        fig_lon = px.histogram(raw_df.sample(min(3000,len(raw_df))), x='pickup_long', nbins=60,
            color_discrete_sequence=['#00e5ff'], template='plotly_dark')
        fig_lon.update_layout(**chart_layout(height=260, title="Longitude", title_font_size=13))
        st.plotly_chart(fig_lon, width="stretch")


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 3: SURGE SIMULATOR                                │
# └─────────────────────────────────────────────────────────┘
elif page == "⚡ Surge Simulator":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Surge Simulator</div>
        <div class="hero-sub">Dynamic pricing engine with multi-horizon forecasting & revenue projection</div>
        <div class="hero-badge">Live Engine · DSR Algorithm · Revenue Analytics</div>
    </div>""", unsafe_allow_html=True)

    if not models_ok:
        st.error("Models not ready. Run `python main.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        fleet = st.number_input("🚗 Fleet", 20, 1000, total_drivers, 10)
    with col2:
        capacity = st.number_input("📦 Capacity/hr", 1.0, 10.0, 3.0, 0.5)
    with col3:
        fare = st.number_input("💰 Base Fare", 5.0, 100.0, avg_fare, 1.0)

    if st.button("⚡ Execute Simulation", type="primary"):
        bar = st.progress(0, "Initializing...")
        for i in range(5):
            bar.progress((i+1)*20, ["Initializing...", "Loading zones...", "Predicting demand...", "Optimizing fleet...", "Computing revenue..."][i])
            time.sleep(0.15)
        alloc_df, rev = run_sim(fleet, sim_hour, day_map[sim_day])
        bar.empty()

        if alloc_df is not None:
            st.markdown("")
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: st.markdown(kpi("💰",f"${rev['Total_Revenue']:,.0f}","Revenue","","up","purple"), unsafe_allow_html=True)
            with c2: st.markdown(kpi("📊",f"{rev['Service_Level']:.1f}%","Service","","up","cyan"), unsafe_allow_html=True)
            with c3: st.markdown(kpi("⚡",f"{rev['Avg_Surge']:.2f}x","Surge","","neutral","amber"), unsafe_allow_html=True)
            with c4: st.markdown(kpi("📈",f"{alloc_df['predicted_demand'].sum():.0f}","Demand","","up","emerald"), unsafe_allow_html=True)
            with c5:
                sh = int((alloc_df['gap']>5).sum()) if 'gap' in alloc_df.columns else 0
                st.markdown(kpi("🔴",f"{sh}","Shortages","","down","rose"), unsafe_allow_html=True)

            st.markdown("")

            # Radar
            st.markdown(section("🎯", "Multi-Horizon Demand Radar", "purple"), unsafe_allow_html=True)
            top = alloc_df.nlargest(10, 'predicted_demand')
            fig_r = go.Figure()
            for horizon, col, color in [('15m','predicted_demand','#7c6aff'),('30m','pred_30m','#00e5ff'),('60m','pred_60m','#ff5c8a')]:
                if col in top.columns:
                    fig_r.add_trace(go.Scatterpolar(
                        r=top[col].values, theta=[f"Z-{z}" for z in top['zone_id']],
                        fill='toself', name=horizon, line=dict(color=color, width=2),
                        fillcolor=color.replace(')', ',0.08)').replace('rgb','rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                    ))
            fig_r.update_layout(**chart_layout(height=440,
                polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(gridcolor='rgba(255,255,255,0.06)')),
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center")))
            st.plotly_chart(fig_r, width="stretch")

            # Gap + Actions
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(section("📊", "Supply-Demand Gap", "rose"), unsafe_allow_html=True)
                gap_vals = alloc_df['gap'].values if 'gap' in alloc_df.columns else np.zeros(len(alloc_df))
                fig_g = go.Figure(go.Bar(
                    x=[f"Z{z}" for z in alloc_df['zone_id']], y=gap_vals,
                    marker_color=['#ff5c8a' if g>0 else '#00e68a' for g in gap_vals],
                    text=[f"{g:+.1f}" for g in gap_vals], textposition='outside',
                    textfont=dict(size=9, family='JetBrains Mono'),
                ))
                fig_g.add_hline(y=0, line_color='rgba(255,255,255,0.15)')
                fig_g.update_layout(**chart_layout(height=360, showlegend=False,
                    yaxis_title="Gap (Demand − Drivers)"))
                st.plotly_chart(fig_g, width="stretch")

            with g2:
                st.markdown(section("🎯", "Action Distribution", "amber"), unsafe_allow_html=True)
                if 'action' in alloc_df.columns:
                    ac = alloc_df['action'].value_counts()
                    fig_d = go.Figure(go.Pie(
                        labels=ac.index, values=ac.values, hole=0.6,
                        marker=dict(colors=COLOR_SEQ[:len(ac)]),
                        textfont=dict(size=11, family='Inter'),
                        textinfo='label+percent',
                    ))
                    fig_d.update_layout(**chart_layout(height=360,
                        legend=dict(orientation="h",y=-0.05,x=0.5,xanchor="center"),
                        annotations=[dict(text='Fleet<br>Actions',x=0.5,y=0.5,font_size=13,showarrow=False,font_color='white')]))
                    st.plotly_chart(fig_d, width="stretch")

            # Revenue Treemap
            if 'revenue' in alloc_df.columns:
                st.markdown(section("💰", "Revenue Treemap by Zone", "purple"), unsafe_allow_html=True)
                fig_t = px.treemap(alloc_df, path=['zone_id'], values='revenue',
                    color='surge_multiplier', color_continuous_scale='Plasma')
                fig_t.update_layout(**chart_layout(height=380, margin=dict(l=5,r=5,t=5,b=5)))
                st.plotly_chart(fig_t, width="stretch")
        else:
            st.error("Simulation failed.")


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 4: ML OBSERVATORY                                 │
# └─────────────────────────────────────────────────────────┘
elif page == "🧠 ML Observatory":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">ML Observatory</div>
        <div class="hero-sub">Deep dive into the LightGBM models powering demand intelligence</div>
        <div class="hero-badge">Feature Importance · Correlations · Architecture</div>
    </div>""", unsafe_allow_html=True)

    if not models_ok:
        st.error("Models not loaded. Run `python main.py` first.")
        st.stop()

    # Architecture cards
    st.markdown(section("🏗️", "Model Architecture", "purple"), unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col, (h, label) in zip([c1,c2,c3], [('15m','15 Min'),('30m','30 Min'),('60m','60 Min')]):
        with col:
            mdl = forecaster.models.get(h)
            if mdl:
                nt = mdl.n_estimators_ if hasattr(mdl,'n_estimators_') else mdl.get_params().get('n_estimators','?')
                nl = mdl.get_params().get('num_leaves', '?')
                lr = mdl.get_params().get('learning_rate', '?')
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:2rem; margin-bottom:0.3rem;">🌲</div>
                    <div style="font-size:1.3rem; font-weight:800; color:var(--text-primary);">{label} Horizon</div>
                    <div style="color:var(--text-muted); font-size:0.8rem; margin:0.5rem 0;">LightGBM Regressor</div>
                    <div class="stat-pill"><strong>{nt}</strong> trees</div>
                    <div class="stat-pill"><strong>{nl}</strong> leaves</div>
                    <div class="stat-pill"><strong>{lr}</strong> LR</div>
                </div>""", unsafe_allow_html=True)

    # Feature Importance
    st.markdown(section("📊", "Feature Importance", "cyan"), unsafe_allow_html=True)
    horizon_sel = st.selectbox("Horizon", ['15m','30m','60m'])
    mdl = forecaster.models.get(horizon_sel)
    if mdl:
        imp = pd.DataFrame({'Feature': mdl.feature_name_, 'Importance': mdl.feature_importances_})
        imp = imp.sort_values('Importance', ascending=True).tail(15)
        fig_imp = go.Figure(go.Bar(x=imp['Importance'], y=imp['Feature'], orientation='h',
            marker=dict(color=imp['Importance'], colorscale=[[0,'#1a103d'],[0.5,'#7c6aff'],[1,'#c4b5fd']], line=dict(width=0)),
            text=imp['Importance'].round(0).astype(int), textposition='outside',
            textfont=dict(size=9, family='JetBrains Mono', color='rgba(240,240,245,0.5)'),
        ))
        fig_imp.update_layout(**chart_layout(height=460, margin=dict(l=130,r=40,t=20,b=40),
            xaxis_title="Split Count Importance", showlegend=False))
        st.plotly_chart(fig_imp, width="stretch")

    # Correlations
    if data_ok:
        st.markdown(section("🔗", "Feature Correlation Matrix", "rose"), unsafe_allow_html=True)
        num_cols = ['demand','hour','day_of_week','lag_1','lag_4','rolling_mean_4','target_15m','target_30m','target_60m']
        ac = [c for c in num_cols if c in processed_df.columns]
        corr = processed_df[ac].corr()
        fig_c = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#0d0887'],[0.25,'#7201a8'],[0.5,'#f0f0f5'],[0.75,'#f77f00'],[1,'#fcbf49']],
            zmin=-1, zmax=1, text=np.round(corr.values,2), texttemplate="%{text}",
            textfont=dict(size=9, family='JetBrains Mono'),
        ))
        fig_c.update_layout(**chart_layout(height=460, margin=dict(l=100,r=20,t=20,b=100)))
        st.plotly_chart(fig_c, width="stretch")


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 5: TIME SERIES LAB                                │
# └─────────────────────────────────────────────────────────┘
elif page == "📈 Time Series Lab":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Time Series Lab</div>
        <div class="hero-sub">Temporal patterns, seasonality analysis & zone-level deep dives</div>
        <div class="hero-badge">Hourly · Daily · Weekly Decomposition</div>
    </div>""", unsafe_allow_html=True)

    if not data_ok:
        st.error("No data. Run `python main.py` first.")
        st.stop()

    # Day of Week
    st.markdown(section("📅", "Day-of-Week Pattern", "purple"), unsafe_allow_html=True)
    dow = processed_df.groupby('day_of_week')['demand'].mean().reset_index()
    dow['day_name'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig_dow = go.Figure(go.Bar(
        x=dow['day_name'], y=dow['demand'],
        marker=dict(color=dow['demand'], colorscale=[[0,'#1a103d'],[0.5,'#7c6aff'],[1,'#c4b5fd']]),
        text=dow['demand'].round(2), textposition='outside',
        textfont=dict(size=10, family='JetBrains Mono'),
    ))
    fig_dow.update_layout(**chart_layout(height=320, showlegend=False, yaxis_title="Avg Demand"))
    st.plotly_chart(fig_dow, width="stretch")

    # Hour × Day Heatmap
    st.markdown(section("🕐", "Hour × Day Demand Heatmap", "cyan"), unsafe_allow_html=True)
    hd = processed_df.groupby(['day_of_week','hour'])['demand'].mean().reset_index()
    pivot = hd.pivot(index='day_of_week', columns='hour', values='demand').fillna(0)
    fig_hd = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(h) for h in pivot.columns],
        y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
        colorscale='Inferno', text=np.round(pivot.values,1), texttemplate="%{text}",
        textfont=dict(size=8),
    ))
    fig_hd.update_layout(**chart_layout(height=320, margin=dict(l=60,r=20,t=20,b=50),
        xaxis_title="Hour", yaxis_title="Day"))
    st.plotly_chart(fig_hd, width="stretch")

    # Zone-level time series
    st.markdown(section("📈", "Zone-Level Demand Over Time", "amber"), unsafe_allow_html=True)
    zones = sorted(processed_df['zone_id'].unique(), key=lambda x: int(x))
    sel_zones = st.multiselect("Select zones", zones, default=zones[:3])
    if sel_zones:
        sub = processed_df[processed_df['zone_id'].isin(sel_zones)].copy()
        sub = sub.groupby(['time_bin','zone_id'])['demand'].sum().reset_index()
        # Resample to hourly for smoother lines
        agg = sub.set_index('time_bin').groupby('zone_id').resample('1h')['demand'].sum().reset_index()
        fig_ts = px.line(agg, x='time_bin', y='demand', color='zone_id',
            color_discrete_sequence=COLOR_SEQ, template='plotly_dark')
        fig_ts.update_layout(**chart_layout(height=380, xaxis_title="Time", yaxis_title="Demand",
            legend=dict(orientation="h",y=1.08,x=0.5,xanchor="center")))
        st.plotly_chart(fig_ts, width="stretch")

    # Cumulative demand
    st.markdown(section("📊", "Cumulative Demand Growth", "rose"), unsafe_allow_html=True)
    ts_all = processed_df.groupby('time_bin')['demand'].sum().reset_index().sort_values('time_bin')
    ts_all['cumulative'] = ts_all['demand'].cumsum()
    fig_cum = go.Figure(go.Scatter(
        x=ts_all['time_bin'], y=ts_all['cumulative'],
        mode='lines', fill='tozeroy',
        line=dict(color='#00e5ff', width=2, shape='spline'),
        fillcolor='rgba(0,229,255,0.06)',
    ))
    fig_cum.update_layout(**chart_layout(height=280, xaxis_title="Time", yaxis_title="Cumulative Rides"))
    st.plotly_chart(fig_cum, width="stretch")


# ┌─────────────────────────────────────────────────────────┐
# │  PAGE 6: DATA EXPLORER                                  │
# └─────────────────────────────────────────────────────────┘
elif page == "📋 Data Explorer":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Data Explorer</div>
        <div class="hero-sub">Explore raw & processed datasets powering the intelligence engine</div>
        <div class="hero-badge">168K+ Records · 20+ Features · 3 Month Span</div>
    </div>""", unsafe_allow_html=True)

    if not data_ok:
        st.error("No data found.")
        st.stop()

    # Stats
    st.markdown(section("📊", "Dataset Statistics", "purple"), unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi("📁",f"{len(processed_df):,}","Rows","processed","up","purple"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("📐",f"{len(processed_df.columns)}","Features","columns","up","cyan"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("🗺️",f"{processed_df['zone_id'].nunique()}","Zones","clusters","up","amber"), unsafe_allow_html=True)
    with c4:
        days = (processed_df['time_bin'].max()-processed_df['time_bin'].min()).days
        st.markdown(kpi("📅",f"{days}","Days","time span","up","emerald"), unsafe_allow_html=True)
    with c5:
        if raw_df is not None:
            st.markdown(kpi("🚗",f"{len(raw_df):,}","Raw Rides","generated","up","rose"), unsafe_allow_html=True)

    st.markdown("")

    tab1, tab2 = st.tabs(["📋 Processed Data", "📋 Raw Data"])
    with tab1:
        st.markdown(section("📋", "Processed Demand Data", "cyan"), unsafe_allow_html=True)
        st.dataframe(processed_df.head(200), height=400, use_container_width=True)
        st.markdown(section("📊", "Column Statistics", "purple"), unsafe_allow_html=True)
        st.dataframe(processed_df.describe().T.round(2), use_container_width=True)

    with tab2:
        if raw_df is not None:
            st.markdown(section("📋", "Raw Rides Data", "rose"), unsafe_allow_html=True)
            st.dataframe(raw_df.head(200), height=400, use_container_width=True)

    # Distribution plots
    st.markdown(section("📈", "Feature Distributions", "amber"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(processed_df, x='demand', nbins=60,
            color_discrete_sequence=['#7c6aff'], template='plotly_dark', title='Demand Distribution')
        fig.update_layout(**chart_layout(height=280, title_font_size=13))
        st.plotly_chart(fig, width="stretch")
    with c2:
        zd = processed_df.groupby('zone_id')['demand'].mean().reset_index().sort_values('demand',ascending=False)
        fig2 = px.bar(zd, x='zone_id', y='demand', color='demand',
            color_continuous_scale=[[0,'#1a103d'],[0.5,'#7c6aff'],[1,'#c4b5fd']],
            template='plotly_dark', title='Avg Demand per Zone')
        fig2.update_layout(**chart_layout(height=280, title_font_size=13, showlegend=False))
        st.plotly_chart(fig2, width="stretch")
