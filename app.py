"""
app.py
------
AquaGuard — main Streamlit entry point.
Run with:  streamlit run app.py
"""

import streamlit as st
import sys, os

# Make sure modules/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from modules import (
    generate_dataframe,
    run_isolation_forest, run_random_forest, get_feature_importance,
    get_anomalies, get_alerts_sorted, alert_summary, ALERT_COLORS,
    anomaly_scatter, feature_importance_bar, zone_distribution, label_pie,
    save_csv, load_csv, data_exists, saved_file_info, DEFAULT_PATH,
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AquaGuard", layout="wide")

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #020617, #0f172a); color: #e2e8f0; }
h1, h2, h3 { color: #38bdf8; }

.metric-card {
    padding: 20px; border-radius: 15px; text-align: center;
    box-shadow: 0 0 15px rgba(0,0,0,0.3); transition: 0.3s; color: white;
}
.metric-card:hover { transform: scale(1.05); }

.card-blue   { background: linear-gradient(135deg, #0369a1, #0ea5e9); }
.card-orange { background: linear-gradient(135deg, #c2410c, #f97316); }
.card-red    { background: linear-gradient(135deg, #991b1b, #ef4444); }
.card-purple { background: linear-gradient(135deg, #6b21a8, #a855f7); }

.info-box {
    background: #1e293b; border-left: 4px solid #38bdf8;
    padding: 10px 16px; border-radius: 8px; margin-bottom: 12px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("💧 AquaGuard")

# ── Data source picker ────────────────────────────────────────────────────────
st.sidebar.markdown("### 📂 Data Source")

file_exists = data_exists()
if file_exists:
    info = saved_file_info()
    st.sidebar.success(f"Saved file found\n\n"
                       f"**Rows:** {info['rows']}  \n"
                       f"**Size:** {info['size_kb']} KB  \n"
                       f"**Saved:** {info['modified']}")
    data_source = st.sidebar.radio(
        "Choose data source",
        ["Load saved CSV", "Generate new data"],
    )
else:
    st.sidebar.info("No saved CSV found — generating new data.")
    data_source = "Generate new data"

num_sensors = 300
if data_source == "Generate new data":
    num_sensors = st.sidebar.slider("Number of sensors", 100, 500, 300)
    if st.sidebar.button("💾 Save generated data"):
        st.session_state["save_requested"] = True

# ── Detection mode ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🤖 Detection Mode")
mode = st.sidebar.radio(
    "Algorithm",
    ["Rule-Based", "Isolation Forest", "Random Forest"],
)

# =========================
# LOAD / GENERATE DATA
# =========================
@st.cache_data(show_spinner="Generating sensor data…")
def fresh_data(n):
    return generate_dataframe(n)

if data_source == "Load saved CSV":
    try:
        df_raw = load_csv()
        st.session_state["df_raw"] = df_raw
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
else:
    df_raw = fresh_data(num_sensors)
    st.session_state["df_raw"] = df_raw
    if st.session_state.pop("save_requested", False):
        path = save_csv(df_raw)
        st.sidebar.success(f"Saved → `{path}`")

df = st.session_state["df_raw"].copy()

# =========================
# APPLY ML MODE
# =========================
if mode == "Isolation Forest":
    df = run_isolation_forest(df)
elif mode == "Random Forest":
    df = run_random_forest(df)

# =========================
# KPIs
# =========================
st.title("💧 AquaGuard Dashboard")

anomalies = get_anomalies(df)
summary   = alert_summary(df)

metrics = [
    ("Total Sensors", len(df),                                       "card-blue"),
    ("Anomalies",     len(anomalies),                                 "card-orange"),
    ("Major Burst",   summary.get("Major_Burst", 0),                  "card-red"),
    ("High Pressure", summary.get("High_Pressure", 0),                "card-purple"),
]

cols = st.columns(4)
for col, (label, val, css) in zip(cols, metrics):
    with col:
        st.markdown(
            f"<div class='metric-card {css}'><h3>{val}</h3><p>{label}</p></div>",
            unsafe_allow_html=True,
        )

# =========================
# ACTIVE ALERTS
# =========================
st.markdown("## 🚨 Active Alerts")

alerts = get_alerts_sorted(df, top_n=8)

if alerts.empty:
    st.success("✅ No active alerts — all sensors normal.")
else:
    for _, row in alerts.iterrows():
        grad, icon = ALERT_COLORS.get(row["active_label"], ("90deg,#333,#555", "⚠️"))
        st.markdown(f"""
        <div style="background:linear-gradient({grad});padding:12px;border-radius:10px;
                    margin-bottom:10px;color:white;">
        {icon} <b>{row['active_label']}</b> |
        Sensor: <b>{row['sensor_id']}</b> |
        Zone: <b>{row['location_zone']}</b> |
        Flow: <b>{row['flow_rate_lpm']:.2f} lpm</b>
        (baseline: {row['baseline_mean']:.2f}) |
        Pressure: <b>{row['pressure_psi']:.2f} psi</b>
        </div>""", unsafe_allow_html=True)

# =========================
# ML INSIGHTS
# =========================
st.markdown("## 🧠 ML Insights")

c1, c2 = st.columns(2)
with c1:
    st.pyplot(anomaly_scatter(df))
with c2:
    importance = get_feature_importance(df)
    st.pyplot(feature_importance_bar(importance))

st.markdown("### 📊 Additional Charts")
c3, c4 = st.columns(2)
with c3:
    st.pyplot(zone_distribution(df))
with c4:
    st.pyplot(label_pie(df))

# =========================
# DATA TABLE + DOWNLOAD
# =========================
st.subheader("📋 Data Table")

show_only = st.checkbox("Show only anomalies")
st.dataframe(anomalies if show_only else df, use_container_width=True)

csv = anomalies.to_csv(index=False).encode()
st.download_button("⬇️ Download Anomalies CSV", csv, "anomalies.csv", "text/csv")
