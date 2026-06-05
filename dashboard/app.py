import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
import time

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Monitor · Hyderabad, Sindh",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

:root {
    --bg:         #f0f4f9;
    --surface:    #ffffff;
    --surface2:   #f7f9fc;
    --border:     #e2e8f2;
    --border2:    #d0d9ea;
    --text:       #0f172a;
    --text2:      #475569;
    --text3:      #94a3b8;
    --accent:     #2563eb;
    --accent2:    #1d4ed8;
    --green:      #16a34a;
    --yellow:     #ca8a04;
    --orange:     #ea580c;
    --red:        #dc2626;
    --purple:     #7c3aed;
    --maroon:     #9f1239;
    --radius:     16px;
    --shadow:     0 2px 12px rgba(15,23,42,0.07);
    --shadow-lg:  0 8px 40px rgba(15,23,42,0.12);
    --card-h:     380px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp { background: var(--bg); }

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1400px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 6px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text3) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1.1rem;
    margin-top: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    /* FIX 1 — match height with weather card */
    min-height: var(--card-h);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

/* ── Blinking dot ── */
.live-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 7px;
    position: relative; top: 1px;
    animation: livepulse 1.8s ease-in-out infinite;
}
@keyframes livepulse {
    0%, 100% { opacity:1; box-shadow: 0 0 0 0 rgba(22,163,74,0.5); }
    50%       { opacity:0.7; box-shadow: 0 0 0 6px rgba(22,163,74,0); }
}

/* ── AQI scale bar ── */
.scale-wrap { margin-top: 1rem; }
.scale-bar {
    height: 8px; border-radius: 8px;
    background: linear-gradient(to right,
        #16a34a 0%,   #16a34a 16.6%,
        #ca8a04 16.6%,#ca8a04 33.3%,
        #ea580c 33.3%,#ea580c 50%,
        #dc2626 50%,  #dc2626 66.6%,
        #7c3aed 66.6%,#7c3aed 83.3%,
        #9f1239 83.3%,#9f1239 100%
    );
    position: relative; margin-bottom: 0.3rem;
}
.scale-labels {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.56rem; color: var(--text3);
    letter-spacing: 0.04em; margin-top: 4px;
}
.scale-marker {
    position: absolute; width: 3px; height: 14px;
    background: var(--text); border-radius: 2px;
    top: -3px; transform: translateX(-50%);
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
}

/* ── Weather card ── */
.weather-card {
    background: linear-gradient(145deg, #1d4ed8 0%, #2563eb 60%, #3b82f6 100%);
    border-radius: var(--radius);
    padding: 1.6rem;
    color: #fff;
    box-shadow: 0 8px 32px rgba(37,99,235,0.3);
    /* FIX 1 — match height with AQI card */
    min-height: var(--card-h);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.weather-temp {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem; font-weight: 800;
    line-height: 1; color: #fff;
}
.weather-desc {
    font-size: 0.95rem; font-weight: 500;
    color: rgba(255,255,255,0.8);
    margin-top: 0.2rem; text-transform: capitalize;
}
.weather-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.8rem; margin-top: 1.2rem;
}
.w-stat {
    background: rgba(255,255,255,0.15);
    border-radius: 10px; padding: 0.65rem 0.8rem;
    backdrop-filter: blur(4px);
}
.w-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.56rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(255,255,255,0.6);
}
.w-stat-val {
    font-size: 1.05rem; font-weight: 700;
    color: #fff; margin-top: 0.1rem;
}

/* ── Pollutant cards ── */
.poll-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.2rem 1rem;
    text-align: center; box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.poll-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-3px); }
.poll-card .poll-top {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: var(--radius) var(--radius) 0 0;
}
.poll-card .poll-name {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text3); margin-top: 0.3rem;
}
.poll-card .poll-val { font-size: 1.8rem; font-weight: 700; color: var(--text); margin: 0.2rem 0; }
.poll-card .poll-unit { font-family: 'DM Mono', monospace; font-size: 0.56rem; color: var(--text3); }
.poll-card .poll-cat {
    font-size: 0.72rem; font-weight: 600; margin-top: 0.3rem;
    padding: 0.2rem 0.6rem; border-radius: 20px; display: inline-block;
}

/* ── Forecast day cards ── */
.fc-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 2rem 1.2rem;
    text-align: center; box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.fc-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-4px); }
.fc-card .fc-top {
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px; border-radius: var(--radius) var(--radius) 0 0;
}
.fc-card .fc-day {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 1.2rem;
}
.fc-card .fc-aqi {
    font-family: 'Playfair Display', serif;
    font-size: 4rem; font-weight: 800; line-height: 1;
}
.fc-card .fc-cat { font-size: 1rem; font-weight: 700; margin-top: 0.6rem; }

/* ── Model cards ── */
.model-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.3rem 1.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    border-left: 3px solid var(--accent);
}
.model-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }

/* ── FIX 3 — Recommendation cards equal height ── */
.rec-col-wrap {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    /* Equal height fix */
    display: flex;
    flex-direction: column;
    height: 200px;              /* fixed height so all 6 cards match */
    box-sizing: border-box;
}
.rec-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-3px); }
.rec-icon { font-size: 1.6rem; margin-bottom: 0.5rem; flex-shrink: 0; }
.rec-title {
    font-size: 0.88rem; font-weight: 700;
    color: var(--text); margin-bottom: 0.4rem;
    flex-shrink: 0;
}
.rec-body {
    font-size: 0.75rem; color: var(--text2);
    line-height: 1.55;
    /* overflow hidden so long text doesn't break layout */
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    flex: 1;
}

/* ── Alert box ── */
.alert-box {
    background: #fff5f5; border: 1px solid #fecaca;
    border-left: 4px solid var(--red);
    border-radius: 10px; padding: 0.9rem 1.3rem;
    margin-bottom: 0.7rem; font-size: 0.86rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
}

hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

/* ── Loading ── */
.loading-step { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--text3); letter-spacing: 0.1em; text-transform: uppercase; }
.loading-step.active { color: var(--accent); }
.loading-step.done   { color: var(--green); }
</style>
""", unsafe_allow_html=True)


# ── AQI helpers ───────────────────────────────────────────────
AQI_LEVELS = [
    (50,  "Good",                 "#16a34a", "#f0fdf4", "#dcfce7"),
    (100, "Moderate",             "#ca8a04", "#fefce8", "#fef9c3"),
    (150, "Unhealthy for Groups", "#ea580c", "#fff7ed", "#ffedd5"),
    (200, "Unhealthy",            "#dc2626", "#fef2f2", "#fee2e2"),
    (300, "Very Unhealthy",       "#7c3aed", "#f5f3ff", "#ede9fe"),
    (999, "Hazardous",            "#9f1239", "#fff1f2", "#ffe4e6"),
]

AQI_ADVICE = {
    "Good":                 "Air quality is satisfactory. Great for outdoor activities.",
    "Moderate":             "Acceptable quality. Unusually sensitive people may be mildly affected.",
    "Unhealthy for Groups": "Sensitive groups should limit prolonged outdoor exertion.",
    "Unhealthy":            "Everyone should reduce prolonged or heavy outdoor activity.",
    "Very Unhealthy":       "Everyone should avoid outdoor activity. Stay indoors if possible.",
    "Hazardous":            "Health emergency — everyone must remain indoors with windows closed.",
}

def aqi_meta(val: float) -> dict:
    for threshold, cat, color, bg, badge in AQI_LEVELS:
        if val <= threshold:
            return {"category": cat, "color": color, "bg": bg, "badge": badge}
    return {"category": "Hazardous", "color": "#9f1239", "bg": "#fff1f2", "badge": "#ffe4e6"}

def aqi_pct(val: float) -> float:
    return min(100, max(0, val / 300 * 100))


# ── MongoDB connection ────────────────────────────────────────
@st.cache_resource
def get_db():
    client = MongoClient(st.secrets["MONGO_URI"], serverSelectionTimeoutMS=8000)
    return client[st.secrets["MONGO_DB"]]


# ── Data loaders ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_latest_prediction():
    db  = get_db()
    doc = db["predictions"].find_one(
        sort=[("made_at", -1)],
        projection={"_id": 0, "aqi_now": 1, "aqi_24h": 1,
                    "aqi_48h": 1, "aqi_72h": 1, "made_at": 1, "timestamp": 1}
    )
    return doc or {}

@st.cache_data(ttl=300)
def load_latest_features():
    db  = get_db()
    doc = db[st.secrets["MONGO_COLLECTION"]].find_one(
        sort=[("timestamp", -1)],
        projection={"_id": 0, "timestamp": 1, "aqi": 1,
                    "pm2_5": 1, "pm10": 1, "no2": 1, "o3": 1,
                    "so2": 1, "co": 1, "humidity": 1, "pressure": 1,
                    "wind_speed": 1, "temperature": 1}
    )
    return doc or {}

@st.cache_data(ttl=300)
def load_trend():
    from datetime import datetime, timedelta, timezone
    db      = get_db()
    cutoff  = datetime.now(timezone.utc) - timedelta(days=7)
    cursor  = db[st.secrets["MONGO_COLLECTION"]].find(
        {"timestamp": {"$gte": cutoff}},
        {"_id": 0, "timestamp": 1, "aqi": 1}
    ).sort("timestamp", 1)   # oldest first — no need to reverse
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


@st.cache_data(ttl=3600)
def load_best_models():
    db     = get_db()
    cursor = db["models"].find(
        {"is_best": True},
        {"_id": 0, "target": 1, "model_name": 1, "metrics": 1, "saved_at": 1}
    )
    return list(cursor)


# ── Trend chart ───────────────────────────────────────────────
def make_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines+markers", name="AQI",
        line=dict(color="#2563eb", width=2.5, shape="spline"),
        marker=dict(size=3.5, color="#2563eb", line=dict(width=1, color="#fff")),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        hovertemplate="<b>%{x|%a %b %d %H:%M}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>"
    ))

    for y0, y1, color in [(0,50,"rgba(22,163,74,0.04)"), (50,100,"rgba(202,138,4,0.04)"),
                           (100,150,"rgba(234,88,12,0.04)"), (150,200,"rgba(220,38,38,0.04)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

    for level, label, color in [(50,"Good","#16a34a"), (100,"Moderate","#ca8a04"), (150,"Unhealthy","#dc2626")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1.2,
                      opacity=0.5, annotation_text=label, annotation_position="right",
                      annotation_font={"size": 9, "color": color})

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafbfd",
        height=280, margin=dict(l=10, r=90, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f2",
                   tickfont={"size": 9, "color": "#94a3b8", "family": "DM Mono"},
                   showline=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f2",
                   tickfont={"size": 9, "color": "#94a3b8", "family": "DM Mono"},
                   title=dict(text="AQI", font={"size": 10, "color": "#94a3b8"}),
                   zeroline=False, range=[0, None]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fff", bordercolor="#2563eb",
                        font={"size": 12, "color": "#0f172a", "family": "DM Sans"}),
        showlegend=False,
    )
    return fig


# ── Loading screen ────────────────────────────────────────────
def show_loading():
    steps = [
        "Connecting to database...",
        "Fetching latest predictions...",
        "Loading historical trend...",
        "Reading model metrics...",
        "Rendering dashboard...",
    ]
    placeholder = st.empty()
    for i, step in enumerate(steps):
        rows = ""
        for j, s in enumerate(steps):
            if j < i:      cls, icon = "done",   "✓"
            elif j == i:   cls, icon = "active", "›"
            else:          cls, icon = "",       "·"
            rows += f'<div class="loading-step {cls}">{icon} {s}</div>'

        pct = int((i + 1) / len(steps) * 100)
        placeholder.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:60vh;gap:1.2rem">
            <div style="font-family:'Playfair Display',serif;font-size:2rem;
                        font-weight:800;color:#0f172a;letter-spacing:-0.02em">
                AQI Monitor
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                        color:#94a3b8;letter-spacing:0.2em">
                HYDERABAD · SINDH · PAKISTAN
            </div>
            <div style="display:flex;flex-direction:column;gap:0.55rem;
                        margin-top:1.5rem;min-width:300px">
                {rows}
            </div>
            <div style="width:280px;height:3px;background:#e2e8f2;
                        border-radius:3px;margin-top:0.8rem;overflow:hidden">
                <div style="width:{pct}%;height:100%;
                            background:linear-gradient(90deg,#2563eb,#16a34a);
                            border-radius:3px;transition:width 0.4s ease"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.45)
    placeholder.empty()


# ── Pollutant card ────────────────────────────────────────────
def pollutant_card(name: str, val, unit: str, color: str, cat: str, badge_bg: str):
    val_str = f"{float(val):.1f}" if val is not None else "—"
    st.markdown(f"""
    <div class="poll-card">
        <div class="poll-top" style="background:{color}"></div>
        <div class="poll-name">{name}</div>
        <div class="poll-val">{val_str}</div>
        <div class="poll-unit">{unit}</div>
        <span class="poll-cat" style="color:{color};background:{badge_bg}">{cat}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Forecast card ─────────────────────────────────────────────
def forecast_card(day_label: str, pred: float):
    if not pred or pred <= 0:
        st.markdown(f"""
        <div class="fc-card">
            <div class="fc-top" style="background:#e2e8f2"></div>
            <div class="fc-day">{day_label}</div>
            <div style="color:#94a3b8;font-size:0.85rem;padding:1.5rem 0">No prediction</div>
        </div>
        """, unsafe_allow_html=True)
        return
    meta = aqi_meta(pred)
    st.markdown(f"""
    <div class="fc-card">
        <div class="fc-top" style="background:{meta['color']}"></div>
        <div class="fc-day">{day_label}</div>
        <div class="fc-aqi" style="color:{meta['color']}">{pred:.0f}</div>
        <div class="fc-cat" style="color:{meta['color']}">{meta['category']}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Recommendation card ───────────────────────────────────────
def rec_card(icon: str, title: str, body: str):
    """Fixed-height recommendation card — body truncated to 4 lines via CSS."""
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-icon">{icon}</div>
        <div class="rec-title">{title}</div>
        <div class="rec-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Pollutant category helpers ────────────────────────────────
def pm25_cat(v):
    if v <= 12:   return "Good",              "#16a34a", "#dcfce7"
    if v <= 35.4: return "Moderate",          "#ca8a04", "#fef9c3"
    if v <= 55.4: return "Unhealthy for SGs", "#ea580c", "#ffedd5"
    return "Unhealthy", "#dc2626", "#fee2e2"

def pm10_cat(v):
    if v <= 54:  return "Good",     "#16a34a", "#dcfce7"
    if v <= 154: return "Moderate", "#ca8a04", "#fef9c3"
    return "Unhealthy", "#dc2626", "#fee2e2"

def generic_cat(v, thresholds):
    for t, cat, col, bg in thresholds:
        if v <= t:
            return cat, col, bg
    return "Hazardous", "#9f1239", "#ffe4e6"


# ── Recommendations data ──────────────────────────────────────
RECOMMENDATIONS = [
    ("😷", "Wear a Mask Outdoors",
     "Always carry an N95 or KN95 mask outdoors, especially near traffic-heavy areas like Autoban Road and Qasimabad."),
    ("🏠", "Keep Windows Closed",
     "During peak pollution hours (7–10 AM and 6–9 PM), keep windows closed and use fans for indoor circulation."),
    ("🌿", "Indoor Plants Help",
     "Spider plants, peace lilies, and snake plants naturally filter particulate matter indoors."),
    ("🚶", "Best Time to Exercise",
     "Early morning (5–6 AM) has lower AQI. Avoid outdoor exercise when AQI exceeds 100."),
    ("👶", "Protect Children & Elderly",
     "Keep vulnerable residents indoors when AQI is above 100 and ensure they stay well-hydrated."),
    ("💧", "Stay Hydrated",
     "Drink 8–10 glasses of water daily. Hydration helps your lungs cope with airborne pollutants."),
]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():

    # Loading screen — first visit only
    if "loaded" not in st.session_state:
        show_loading()
        st.session_state["loaded"] = True

    # ── Fetch data ────────────────────────────────────────────
    try:
        latest   = load_latest_prediction()
        features = load_latest_features()
        trend_df = load_trend()
        models   = load_best_models()
    except Exception as e:
        st.error(f"⚠️ Could not connect to database: {e}")
        st.info("Make sure MONGO_URI is set in .streamlit/secrets.toml")
        return

    if not latest:
        st.warning("No prediction data found. Run the hourly pipeline first.")
        return

    current_aqi  = float(latest.get("aqi_now", 0))
    current_meta = aqi_meta(current_aqi)

    # ── Header ────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        try:
            ts = pd.to_datetime(latest.get("made_at", "")).strftime("%B %d, %Y · %H:%M PKT")
        except Exception:
            ts = datetime.now().strftime("%B %d, %Y · %H:%M PKT")

        st.markdown(f"""
        <div style="margin-bottom:0.2rem">
            <span class="live-dot"></span>
            <span style="font-family:'DM Mono',monospace;font-size:0.62rem;
                         letter-spacing:0.14em;color:#16a34a;text-transform:uppercase">Live</span>
        </div>
        <h1 style="font-family:'Playfair Display',serif;font-weight:800;
                   font-size:2.5rem;color:#0f172a;margin:0.2rem 0 0;line-height:1.1">
            Hyderabad, Sindh
            <span style="color:#2563eb"> Air Quality</span>
        </h1>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                    color:#94a3b8;letter-spacing:0.12em;margin-top:0.5rem">
            Last updated: {ts}
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("""
        <div style="text-align:right;padding-top:1.5rem">
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                        color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase">Powered by</div>
            <div style="font-size:0.82rem;font-weight:600;color:#0f172a;margin-top:0.1rem">
                MongoDB · GitHub Actions · Streamlit
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — CURRENT AQI + WEATHER
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">Current Air Quality</div>', unsafe_allow_html=True)

    col_aqi, col_wx = st.columns([1.6, 1], gap="large")

    with col_aqi:
        pct_pos = aqi_pct(current_aqi)
        # FIX 1 — PM pills removed, padding/height via CSS class
        st.markdown(f"""
<div class="card" style="border-left:4px solid {current_meta['color']}">
    <div>
        <div style="display:flex;align-items:flex-end;gap:0.8rem">
            <div style="font-family:'Playfair Display',serif;font-size:5.5rem;
                        font-weight:800;color:{current_meta['color']};line-height:1">
                {current_aqi:.0f}
            </div>
            <div style="padding-bottom:0.6rem">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                            color:#94a3b8;letter-spacing:0.12em;text-transform:uppercase">
                    AQI (US)
                </div>
                <div style="font-size:1.2rem;font-weight:700;color:{current_meta['color']};margin-top:0.1rem">
                    Air Quality is
                </div>
                <div style="font-size:1.5rem;font-weight:800;color:{current_meta['color']}">
                    {current_meta['category']}
                </div>
            </div>
        </div>
        <div class="scale-wrap">
            <div class="scale-bar">
                <div class="scale-marker" style="left:{pct_pos:.1f}%"></div>
            </div>
            <div class="scale-labels">
                <span>0 Good</span><span>50</span><span>100</span>
                <span>150</span><span>200</span><span>300 Hazardous</span>
            </div>
        </div>
    </div>
    <div style="margin-top:1.4rem;
                background:{current_meta['bg']};
                border-left:3px solid {current_meta['color']};
                border-radius:8px;
                padding:0.9rem 1rem;
                font-size:0.84rem;
                color:#475569;
                line-height:1.6">
        {AQI_ADVICE.get(current_meta['category'], '')}
    </div>
</div>
""", unsafe_allow_html=True)

    with col_wx:
        temp     = features.get("temperature")
        humidity = features.get("humidity")
        pressure = features.get("pressure")
        wind     = features.get("wind_speed")

        temp_str  = f"{float(temp):.0f}°C"       if temp     is not None else "—"
        hum_str   = f"{float(humidity):.0f}%"    if humidity is not None else "—"
        press_str = f"{float(pressure):.0f} hPa" if pressure is not None else "—"
        wind_str  = f"{float(wind):.1f} km/h"    if wind     is not None else "—"

        st.markdown(f"""
<div class="weather-card">
    <div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.18em;color:rgba(255,255,255,0.6);
                    text-transform:uppercase;margin-bottom:0.5rem">
            Weather Conditions
        </div>
        <div class="weather-temp">{temp_str}</div>
        <div class="weather-desc">Hyderabad, Sindh, PK</div>
    </div>
    <div class="weather-grid">
        <div class="w-stat">
            <div class="w-stat-label">💧 Humidity</div>
            <div class="w-stat-val">{hum_str}</div>
        </div>
        <div class="w-stat">
            <div class="w-stat-label">🌡 Pressure</div>
            <div class="w-stat-val">{press_str}</div>
        </div>
        <div class="w-stat">
            <div class="w-stat-label">💨 Wind Speed</div>
            <div class="w-stat-val">{wind_str}</div>
        </div>
        <div class="w-stat">
            <div class="w-stat-label">📍 Location</div>
            <div class="w-stat-val" style="font-size:0.78rem">25.37°N 68.37°E</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — MAJOR POLLUTANTS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">Major Air Pollutants</div>', unsafe_allow_html=True)

    pm25 = float(features.get("pm2_5", 0) or 0)
    pm10 = float(features.get("pm10",  0) or 0)
    no2  = float(features.get("no2",   0) or 0)
    o3   = float(features.get("o3",    0) or 0)
    so2  = float(features.get("so2",   0) or 0)
    co   = float(features.get("co",    0) or 0)

    no2_thresh = [(53,"Good","#16a34a","#dcfce7"),(100,"Moderate","#ca8a04","#fef9c3"),(360,"Unhealthy","#dc2626","#fee2e2")]
    o3_thresh  = [(54,"Good","#16a34a","#dcfce7"),(70,"Moderate","#ca8a04","#fef9c3"),(85,"Unhealthy for SGs","#ea580c","#ffedd5")]
    so2_thresh = [(35,"Good","#16a34a","#dcfce7"),(75,"Moderate","#ca8a04","#fef9c3"),(185,"Unhealthy","#dc2626","#fee2e2")]
    co_thresh  = [(4.4,"Good","#16a34a","#dcfce7"),(9.4,"Moderate","#ca8a04","#fef9c3"),(12.4,"Unhealthy for SGs","#ea580c","#ffedd5")]

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    pollutants = [
        (p1, "PM2.5", pm25, "µg/m³", *pm25_cat(pm25)),
        (p2, "PM10",  pm10, "µg/m³", *pm10_cat(pm10)),
        (p3, "NO₂",   no2,  "ppb",   *generic_cat(no2, no2_thresh)),
        (p4, "O₃",    o3,   "ppb",   *generic_cat(o3,  o3_thresh)),
        (p5, "SO₂",   so2,  "ppb",   *generic_cat(so2, so2_thresh)),
        (p6, "CO",    co,   "ppm",   *generic_cat(co,  co_thresh)),
    ]
    for col, name, val, unit, cat, color, badge in pollutants:
        with col:
            pollutant_card(name, val, unit, color, cat, badge)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — 3-DAY FORECAST
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">ML Forecast · Next 3 Days</div>', unsafe_allow_html=True)

    now = datetime.now()
    horizon_keys = [
        ((now + timedelta(days=1)).strftime("%A · %d-%b").upper(), "aqi_24h"),
        ((now + timedelta(days=2)).strftime("%A · %d-%b").upper(), "aqi_48h"),
        ((now + timedelta(days=3)).strftime("%A · %d-%b").upper(), "aqi_72h"),
    ]

    fc1, fc2, fc3 = st.columns(3)
    for col, (day_label, key) in zip([fc1, fc2, fc3], horizon_keys):
        pred = float(latest.get(key, 0) or 0)
        with col:
            forecast_card(day_label, pred)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 4 — HISTORICAL TREND
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">Historical AQI Trend · Last 7 Days</div>',
                unsafe_allow_html=True)

    if not trend_df.empty:
        st.plotly_chart(make_trend_chart(trend_df), use_container_width=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, label, value in [
            (sc1, "7-Day Average", f"{trend_df['aqi'].mean():.0f}"),
            (sc2, "7-Day Max",     f"{trend_df['aqi'].max():.0f}"),
            (sc3, "7-Day Min",     f"{trend_df['aqi'].min():.0f}"),
            (sc4, "Data Points",   f"{len(trend_df)}"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#ffffff;border:1px solid #e2e8f2;border-radius:16px;
                            padding:1.2rem;text-align:center;box-shadow:0 2px 12px rgba(15,23,42,0.07)">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                                letter-spacing:0.16em;text-transform:uppercase;
                                color:#94a3b8;margin-bottom:0.4rem">{label}</div>
                    <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                                font-weight:800;color:#0f172a;line-height:1">{value}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No historical data found in features collection.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 5 — BEST MODELS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">Best ML Models & Performance</div>',
                unsafe_allow_html=True)

    target_labels = {
        "aqi_24h": "24-Hour Forecast",
        "aqi_48h": "48-Hour Forecast",
        "aqi_72h": "72-Hour Forecast",
    }

    if models:
        model_cols = st.columns(min(len(models), 3))
        for idx, m in enumerate(sorted(models, key=lambda x: x.get("target", ""))):
            target     = m.get("target", "—").strip('"')
            model_name = m.get("model_name", "—").replace("_", " ").title()
            metrics    = m.get("metrics", {})
            saved_at   = m.get("saved_at", "—")
            horizon    = target_labels.get(target, target)

            mae  = metrics.get("mae",  metrics.get("MAE",  None))
            rmse = metrics.get("rmse", metrics.get("RMSE", None))
            r2   = metrics.get("r2",   metrics.get("R2",   None))
            mae_s  = f"{float(mae):.2f}"  if mae  is not None else "—"
            rmse_s = f"{float(rmse):.2f}" if rmse is not None else "—"
            r2_s   = f"{float(r2):.3f}"   if r2   is not None else "—"
            try:    saved_str = pd.to_datetime(saved_at).strftime("%b %d, %Y")
            except: saved_str = str(saved_at)[:10]

            with model_cols[idx % 3]:
                st.markdown(f"""
<div class="model-card">
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                letter-spacing:0.16em;text-transform:uppercase;
                color:#2563eb;margin-bottom:0.3rem">{horizon}</div>
    <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.8rem">{model_name}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem">
        <div style="background:#f0f4f9;border-radius:8px;padding:0.55rem 0.4rem;text-align:center">
            <div style="font-family:'DM Mono',monospace;font-size:0.54rem;color:#94a3b8;text-transform:uppercase">MAE</div>
            <div style="font-size:1rem;font-weight:700;color:#0f172a">{mae_s}</div>
        </div>
        <div style="background:#f0f4f9;border-radius:8px;padding:0.55rem 0.4rem;text-align:center">
            <div style="font-family:'DM Mono',monospace;font-size:0.54rem;color:#94a3b8;text-transform:uppercase">RMSE</div>
            <div style="font-size:1rem;font-weight:700;color:#0f172a">{rmse_s}</div>
        </div>
        <div style="background:#f0f9f4;border-radius:8px;padding:0.55rem 0.4rem;text-align:center">
            <div style="font-family:'DM Mono',monospace;font-size:0.54rem;color:#94a3b8;text-transform:uppercase">R²</div>
            <div style="font-size:1rem;font-weight:700;color:#16a34a">{r2_s}</div>
        </div>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#94a3b8;margin-top:0.7rem">
        Trained {saved_str}
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("No model metrics found. Run training pipeline first.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 6 — ALERTS (conditional)
    # ══════════════════════════════════════════════════════════
    alert_keys = {"aqi_24h": "24h Forecast", "aqi_48h": "48h Forecast", "aqi_72h": "72h Forecast"}
    alerts = [(label, float(latest.get(k, 0) or 0))
              for k, label in alert_keys.items()
              if float(latest.get(k, 0) or 0) > 150]

    if alerts:
        st.markdown('<div class="sec-label">⚠ Air Quality Alerts</div>', unsafe_allow_html=True)
        for label, pred in alerts:
            meta = aqi_meta(pred)
            st.markdown(f"""
<div class="alert-box">
    <strong style="color:#dc2626;font-family:'DM Mono',monospace;
                   font-size:0.72rem;letter-spacing:0.1em">{label.upper()} ALERT</strong> —
    AQI <span style="color:{meta['color']};font-weight:700">{pred:.0f}</span>
    <span style="color:{meta['color']}"> ({meta['category']})</span><br>
    <span style="color:#475569;font-size:0.82rem">{AQI_ADVICE.get(meta['category'], '')}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 7 — RECOMMENDATIONS
    # FIX 3 — fixed height cards, body truncated to 4 lines
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="sec-label">Recommendations for Hyderabad Residents</div>',
                unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r4, r5, r6 = st.columns(3)

    for col, (icon, title, body) in zip([r1, r2, r3], RECOMMENDATIONS[:3]):
        with col:
            rec_card(icon, title, body)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    for col, (icon, title, body) in zip([r4, r5, r6], RECOMMENDATIONS[3:]):
        with col:
            rec_card(icon, title, body)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            font-family:'DM Mono',monospace;font-size:0.58rem;
            letter-spacing:0.08em;color:#cbd5e1;padding:0.2rem 0">
    <span>AQI MONITOR · HYDERABAD SINDH · PK</span>
    <span>DATA SOURCE: OPENMETEO</span>
    <span>© {datetime.now().year}</span>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()