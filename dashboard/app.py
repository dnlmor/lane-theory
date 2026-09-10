"""
Lane Theory — High-Performance Dashboard
Streamlit app: input a race, get a High-Performance Diagnostic report —
plain-English, segment-based, grounded entirely in real computed values.

Run with: streamlit run dashboard/app.py
"""

import sys
import os
import re
import joblib
import pickle
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features
from optimization import (
    compute_pillar_report, find_biggest_gap_pillar,
    simulate_primary_gap, get_direction, PILLARS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
INDIVIDUALS_RAW_PATH = os.path.join(DATA_DIR, "individuals", "raw", "individuals_raw.csv")
ELITE_FEATURES_PATH = os.path.join(DATA_DIR, "elite", "processed", "100m_freestyle_features.csv")

RAW_COLUMNS = [
    "athlete_id", "race_id", "name", "gender", "height_cm", "date_of_birth",
    "race_date", "final_time_sec", "pb_50m_seconds", "l1_reaction_time",
    "l1_breakout_distance", "l1_breakout_time", "l1_stroke_count", "l1_total_time",
    "l1_split_25m", "l2_breakout_distance", "l2_breakout_time", "l2_stroke_count",
    "l2_total_time", "l2_split_25m",
]

METRIC_INFO = {
    "pacing_decay_ratio": ("Pacing Decay", "Second half deceleration relative to first half."),
    "intra_lap1_fade_ratio": ("Lap 1 Steadiness", "Speed retained within the first 50m."),
    "intra_lap2_fade_ratio": ("Lap 2 Steadiness", "Speed retained within the second 50m."),
    "finish_vs_fresh_ratio": ("Finishing Strength", "Final 25m pace as share of fresh pace."),
    "breakout_decay_ratio": ("Breakout Retention", "Underwater distance kept on turn wall vs dive."),
    "l1_breakout_pct": ("Lap 1 Underwater Share", "Share of first lap spent underwater."),
    "l2_breakout_pct": ("Lap 2 Underwater Share", "Share of second lap spent underwater."),
    "l1_underwater_speed": ("Lap 1 Underwater Speed", "Speed off dive wall."),
    "l2_underwater_speed": ("Lap 2 Underwater Speed", "Speed off turn wall."),
    "si_retention": ("Stroke Efficiency Retention", "Lap 1 efficiency retained in Lap 2."),
    "l1_relative_stroke_length": ("Lap 1 Stroke Reach", "Distance per stroke in Lap 1."),
    "l2_relative_stroke_length": ("Lap 2 Stroke Reach", "Distance per stroke in Lap 2."),
    "l1_stroke_rate": ("Lap 1 Stroke Rate", "Strokes per minute in Lap 1."),
    "l2_stroke_rate": ("Lap 2 Stroke Rate", "Strokes per minute in Lap 2."),
    "surface_speed_decay_ratio": ("Surface Speed Retention", "Clean swim speed kept in Lap 2."),
    "l1_stroke_efficiency_pct": ("Lap 1 Stroke Efficiency", "Reach relative to optimal average."),
    "l2_stroke_efficiency_pct": ("Lap 2 Stroke Efficiency", "Reach relative to optimal average."),
}

ACTION_TEXT = {
    "l1_underwater_speed": "Build underwater kick power off the dive.",
    "l2_underwater_speed": "Build underwater kick power off the turn wall.",
    "breakout_decay_ratio": "Extend Lap 2 breakout distance closer to Lap 1.",
    "l1_breakout_pct": "Extend underwater phase off dive (within 15m rule).",
    "l2_breakout_pct": "Extend underwater phase off turn wall.",
    "pacing_decay_ratio": "Work on even pacing distribution across laps.",
    "si_retention": "Focus on stroke grip and length as fatigue hits.",
    "l1_relative_stroke_length": "Focus on longer, complete pull in Lap 1.",
    "l2_relative_stroke_length": "Focus on complete pull extension in Lap 2.",
    "l1_stroke_rate": "Optimize Lap 1 cadence to match stroke reach.",
    "l2_stroke_rate": "Avoid over-spinning late — keep leverage over turnover.",
    "surface_speed_decay_ratio": "Maintain clean swimming speed independent of wall push.",
    "intra_lap1_fade_ratio": "Smooth out speed effort within first 50m.",
    "intra_lap2_fade_ratio": "Smooth out speed effort within second 50m.",
    "finish_vs_fresh_ratio": "Build closing speed capacity for final 25m.",
    "l1_stroke_efficiency_pct": "Combine stroke length and power in Lap 1.",
    "l2_stroke_efficiency_pct": "Combine stroke length and power in Lap 2.",
}

VERDICT_STYLE = {
    "Strength": ("#34d399", "rgba(52, 211, 153, 0.12)", "rgba(52, 211, 153, 0.3)", "Strength"),
    "On-par": ("#38bdf8", "rgba(56, 189, 248, 0.12)", "rgba(56, 189, 248, 0.3)", "On-par"),
    "Gap": ("#fbbf24", "rgba(251, 191, 36, 0.12)", "rgba(251, 191, 36, 0.3)", "Gap"),
    "Critical Gap": ("#f87171", "rgba(248, 113, 113, 0.12)", "rgba(248, 113, 113, 0.3)", "Critical Gap"),
}

VERDICT_EXPLAIN = {
    "Strength": "At or above the optimal range — clear performance advantage.",
    "On-par": "Inside optimal range, with room to shift toward the upper threshold.",
    "Gap": "Outside optimal range — identified training focus area.",
    "Critical Gap": "Significantly outside optimal range — highest return opportunity.",
}

SEVERITY_RANK = {"Critical Gap": 0, "Gap": 1, "On-par": 2}

# ---------- Loaders ----------

@st.cache_resource
def load_models():
    return joblib.load(os.path.join(MODELS_DIR, "gender_models.joblib"))

@st.cache_resource
def load_shap_values():
    with open(os.path.join(MODELS_DIR, "shap_values_dict.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_elite_features():
    return pd.read_csv(ELITE_FEATURES_PATH)

def load_individuals_raw():
    if os.path.exists(INDIVIDUALS_RAW_PATH):
        return pd.read_csv(INDIVIDUALS_RAW_PATH)
    return pd.DataFrame(columns=RAW_COLUMNS)

def get_next_athlete_id(existing_df):
    if existing_df.empty:
        return "U001"
    nums = existing_df["athlete_id"].str.extract(r"U(\d+)")[0].astype(int)
    return f"U{nums.max() + 1:03d}"

def get_next_race_id(athlete_id, existing_df):
    athlete_races = existing_df[existing_df["athlete_id"] == athlete_id]
    if athlete_races.empty:
        return f"{athlete_id}-R01"
    nums = athlete_races["race_id"].str.extract(r"-R(\d+)")[0].astype(int)
    return f"{athlete_id}-R{nums.max() + 1:02d}"

def render_html(html):
    flat = re.sub(r"\n\s*", "", html)
    st.markdown(flat, unsafe_allow_html=True)

# ---------- Calculations & Formatting ----------

def compute_severity(metric, value, elite_min, elite_max, elite_mean, verdict, sub):
    if verdict != "Gap":
        return verdict
    direction = get_direction(metric, sub)
    if direction == "higher":
        span = max(elite_mean - elite_min, 1e-9)
        deficit = elite_min - value
    else:
        span = max(elite_max - elite_mean, 1e-9)
        deficit = value - elite_max
    return "Critical Gap" if deficit > span else "Gap"

def format_metric(entry, row, sub):
    m, v, mean = entry["metric"], entry["value"], entry["elite_mean"]

    if m == "pacing_decay_ratio":
        return f"{(v - 1) * 100:+.1f}%", f"Optimal: {(mean - 1) * 100:+.1f}%"

    if m in ("finish_vs_fresh_ratio", "breakout_decay_ratio", "si_retention",
             "surface_speed_decay_ratio", "l1_breakout_pct", "l2_breakout_pct",
             "intra_lap1_fade_ratio", "intra_lap2_fade_ratio"):
        return f"{v * 100:.0f}%", f"Optimal: {mean * 100:.0f}%"

    if m in ("l1_underwater_speed", "l2_underwater_speed"):
        return f"{v:.2f} m/s", f"Optimal: {mean:.2f} m/s"

    if m in ("l1_stroke_rate", "l2_stroke_rate"):
        return f"{v:.0f} spm", f"Optimal: {mean:.0f} spm"

    if m in ("l1_relative_stroke_length", "l2_relative_stroke_length"):
        lap = m[1]
        raw_val = row[f"l{lap}_stroke_length"]
        raw_mean = sub[f"l{lap}_stroke_length"].mean()
        return f"{raw_val:.2f} m", f"Optimal: {raw_mean:.2f} m"

    return f"{v:.3f}", f"Optimal: {mean:.3f}"

def compute_stroke_efficiency(row, elite_df, pillar_report):
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]
    verdict_lookup = {e["metric"]: e["severity"] for entries in pillar_report["pillars"].values() for e in entries}
    result = {}
    for lap in [1, 2]:
        col = f"l{lap}_relative_stroke_length"
        mean = sub[col].mean()
        pct = (row[col] / mean) * 100
        result[f"l{lap}_stroke_efficiency_pct"] = (pct, verdict_lookup.get(col, "On-par"))
    return result

# ---------- Pure Dark Mode Responsive CSS ----------

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background-color: #090d16 !important;
    color: #e2e8f0;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 1350px;
    width: 100%;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Compact Console Wrapper to stop wide stretching */
.console-container {
    max-width: 820px;
    margin: 0 auto;
    width: 100%;
}

.form-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

.form-header {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
    margin-bottom: 10px;
}

/* Diagnostic Dashboard Header */
.dash-hero {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    box-sizing: border-box;
}

.hero-main {
    flex: 1 1 300px;
    min-width: 0;
}

.hero-title-text {
    font-size: clamp(20px, 4vw, 32px);
    font-weight: 800;
    color: #ffffff;
    word-break: break-word;
    margin: 0 0 8px 0;
}

.hero-tag-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.hero-badge {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 12px;
    color: #94a3b8;
    white-space: nowrap;
}

.hero-time-badge {
    flex: 0 1 auto;
    background: #030712;
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 12px 20px;
    text-align: center;
    min-width: 140px;
    box-sizing: border-box;
}

.hero-time-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(24px, 5vw, 36px);
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
}

/* Flexible Grid Card UI */
.tile-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
    box-sizing: border-box;
    width: 100%;
}

.tile-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #f3f4f6;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tile-num-icon {
    background: #0284c7;
    color: #ffffff;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
    flex-shrink: 0;
}

/* Metric Rows */
.metric-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1f2937;
    gap: 8px;
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-info-side {
    flex: 1 1 160px;
    min-width: 0;
}

.metric-name {
    font-size: 13px;
    font-weight: 600;
    color: #f3f4f6;
    word-break: break-word;
}

.metric-sub {
    font-size: 11px;
    color: #6b7280;
}

.metric-val-side {
    flex: 0 0 auto;
    text-align: right;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.metric-value-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    white-space: nowrap;
}

.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    white-space: nowrap;
}

.alert-callout {
    background: #030712;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 10px;
    border-left: 3px solid #38bdf8;
    font-size: 12px;
    color: #cbd5e1;
    line-height: 1.4;
    word-break: break-word;
}

/* Sidebar Panels */
.side-panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-sizing: border-box;
}

.side-panel-header {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 10px;
}

.action-row {
    font-size: 12px;
    color: #d1d5db;
    padding: 8px 0;
    border-bottom: 1px solid #1f2937;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    word-break: break-word;
}

.action-row:last-child {
    border-bottom: none;
}

/* Tight Streamlit Component Inputs */
div[data-baseweb="input"] {
    background-color: #030712 !important;
    border-color: #374151 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    padding: 4px;
    border-radius: 10px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #9ca3af;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #1f2937 !important;
    color: #38bdf8 !important;
}
</style>
    """, unsafe_allow_html=True)

def badge_html(verdict):
    fg, bg, border, label = VERDICT_STYLE[verdict]
    return f"<span class='status-badge' style='color:{fg}; background:{bg}; border: 1px solid {border};'>{label}</span>"

def stat_row_html(label, value_str, caption_str, verdict):
    return (f'<div class="metric-row"><div class="metric-info-side"><div class="metric-name">{label}</div>'
            f'<div class="metric-sub">{caption_str}</div></div>'
            f'<div class="metric-val-side"><span class="metric-value-text">{value_str}</span>{badge_html(verdict)}</div></div>')

def render_tile_segment(num, title, entries, row, sub, extra_stats_html="", positive_msg="", action_extra=""):
    severities = [e["severity"] for e in entries]
    color = "#f87171" if "Critical Gap" in severities else ("#fbbf24" if "Gap" in severities else "#38bdf8")

    stats_html = ""
    for e in entries:
        label, _ = METRIC_INFO.get(e["metric"], (e["metric"], ""))
        value_str, caption_str = format_metric(e, row, sub)
        stats_html += stat_row_html(label, value_str, caption_str, e["severity"])
    stats_html += extra_stats_html

    gaps = [e["metric"] for e in entries if e["severity"] in ("Gap", "Critical Gap")]
    diag_text = (" ".join(ACTION_TEXT.get(g, "") for g in gaps) + (" " + action_extra if action_extra else "")) if gaps else positive_msg

    render_html(f'''
    <div class="tile-card">
        <div class="tile-title"><span class="tile-num-icon">{num}</span>{title}</div>
        {stats_html}
        <div class="alert-callout" style="border-left-color:{color};">
            <b>Diagnostic:</b> {diag_text}
        </div>
    </div>
    ''')

# ---------- Main Application Setup ----------

st.set_page_config(page_title="Lane Theory Dashboard", layout="wide", page_icon="🏊")
inject_css()

elite_features = load_elite_features()
models = load_models()
shap_values_dict = load_shap_values()
individuals_raw = load_individuals_raw()

st.title("🏊 Lane Theory")
st.caption("High-Performance Swimming Diagnostic Engine — Powered by Relative Elite Analytics")

tab_new, tab_existing = st.tabs(["📝 Race Input Console", "📂 Saved Race Records"])

with tab_new:
    # Centered compact wrapper for clean laptop view
    st.markdown('<div class="console-container">', unsafe_allow_html=True)

    st.markdown('<div class="form-card"><div class="form-header">1. Swimmer Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name", value="Daniel Siahaan", key="in_name")
        gender = st.selectbox("Gender", ["male", "female"], key="in_gender")
    with c2:
        height_cm = st.number_input("Height (cm)", 100.0, 230.0, 180.0, key="in_height")
        dob = st.date_input("Date of Birth", value=date(2002, 5, 15), key="in_dob")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-card"><div class="form-header">2. Overall Race Totals</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        final_time_sec = st.number_input("Final Time (s)", 40.0, 180.0, 52.40, key="in_final")
    with r2:
        pb_50m_seconds = st.number_input("50m PB (s)", 18.0, 60.0, 23.80, key="in_pb50")
    with r3:
        l1_reaction_time = st.number_input("Reaction Time (s)", 0.3, 1.5, 0.65, key="in_rt")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-card"><div class="form-header">3. Lap Splits & Segment Data</div>', unsafe_allow_html=True)
    lap1_col, lap2_col = st.columns(2)
    
    with lap1_col:
        st.caption("Lap 1 (0–50m)")
        l1_split_25m = st.number_input("25m Split (s)", 8.0, 40.0, 11.20, key="l1s25")
        l1_total_time = st.number_input("Lap 1 Total Time (s)", 10.0, 60.0, 25.10, key="l1tt")
        l1_breakout_distance = st.number_input("Breakout Distance (m)", 0.0, 15.0, 11.5, key="l1bd")
        l1_breakout_time = st.number_input("Breakout Time (s)", 0.5, 10.0, 4.20, key="l1bt")
        l1_stroke_count = st.number_input("Lap 1 Stroke Count", 1, 60, 34, key="l1sc")

    with lap2_col:
        st.caption("Lap 2 (50–100m)")
        l2_split_25m = st.number_input("75m Split (Lap Relative s)", 8.0, 40.0, 13.50, key="l2s25")
        l2_total_time = final_time_sec - l1_total_time
        st.text_input("Lap 2 Total Time (Auto s)", value=f"{l2_total_time:.2f}", disabled=True)
        l2_breakout_distance = st.number_input("Breakout Distance (m)", 0.0, 15.0, 7.20, key="l2bd")
        l2_breakout_time = st.number_input("Breakout Time (s)", 0.5, 10.0, 3.10, key="l2bt")
        l2_stroke_count = st.number_input("Lap 2 Stroke Count", 1, 60, 38, key="l2sc")
    st.markdown('</div>', unsafe_allow_html=True)

    race_date_input = date.today()
    run_new = st.button("🚀 Run Performance Diagnosis", key="run_new", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if run_new:
        athlete_id = get_next_athlete_id(individuals_raw)
        race_id = get_next_race_id(athlete_id, individuals_raw)
        new_row = {
            "athlete_id": athlete_id, "race_id": race_id, "name": name, "gender": gender,
            "height_cm": height_cm, "date_of_birth": dob.isoformat(),
            "race_date": race_date_input.isoformat(), "final_time_sec": final_time_sec,
            "pb_50m_seconds": pb_50m_seconds, "l1_reaction_time": l1_reaction_time,
            "l1_breakout_distance": l1_breakout_distance, "l1_breakout_time": l1_breakout_time,
            "l1_stroke_count": l1_stroke_count, "l1_total_time": l1_total_time,
            "l1_split_25m": l1_split_25m,
            "l2_breakout_distance": l2_breakout_distance, "l2_breakout_time": l2_breakout_time,
            "l2_stroke_count": l2_stroke_count, "l2_total_time": l2_total_time,
            "l2_split_25m": l2_split_25m,
        }
        individuals_raw = pd.concat([individuals_raw, pd.DataFrame([new_row])], ignore_index=True)
        os.makedirs(os.path.dirname(INDIVIDUALS_RAW_PATH), exist_ok=True)
        individuals_raw.to_csv(INDIVIDUALS_RAW_PATH, index=False)

        row_df = engineer_features(pd.DataFrame([new_row]))
        st.session_state["diagnosis_row"] = row_df.iloc[0]

with tab_existing:
    st.markdown('<div class="console-container">', unsafe_allow_html=True)
    if individuals_raw.empty:
        st.info("No stored race logs found.")
    else:
        df = individuals_raw.copy()
        df["occurrence"] = (df.groupby("name").cumcount() + 1).astype(int)
        df["display_label"] = df.apply(
            lambda r: f"{r['name']} — {r['final_time_sec']:.2f}s (Race #{int(r['occurrence'])})", axis=1
        )
        label_to_race_id = dict(zip(df["display_label"], df["race_id"]))
        chosen_label = st.selectbox("Select Race Record", df["display_label"].tolist())
        if st.button("📂 Load Selected Race Data", key="run_existing", use_container_width=True, type="primary"):
            chosen_race_id = label_to_race_id[chosen_label]
            selected_raw = individuals_raw[individuals_raw["race_id"] == chosen_race_id]
            row_df = engineer_features(selected_raw)
            st.session_state["diagnosis_row"] = row_df.iloc[0]
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================================
# DASHBOARD REPORT LAYOUT
# =======================================================================

if "diagnosis_row" in st.session_state:
    row = st.session_state["diagnosis_row"]
    gender = row["gender"]
    sub = elite_features[elite_features["gender"] == gender]

    pillar_report = compute_pillar_report(row, elite_features)
    biggest_gap_pillars = find_biggest_gap_pillar(pillar_report)

    for pillar, entries in pillar_report["pillars"].items():
        for e in entries:
            e["severity"] = compute_severity(e["metric"], e["value"], e["elite_min"],
                                              e["elite_max"], e["elite_mean"], e["verdict"], sub)

    stroke_eff = compute_stroke_efficiency(row, elite_features, pillar_report)

    # 1. Full-Width Responsive Hero Banner
    render_html(f'''
    <div class="dash-hero">
        <div class="hero-main">
            <div class="hero-title-text">{row['name']}</div>
            <div class="hero-tag-group">
                <span class="hero-badge">👤 {gender.capitalize()}</span>
                <span class="hero-badge">📏 {row['height_cm']:.0f} cm</span>
                <span class="hero-badge">🏊 100m Freestyle LCM</span>
            </div>
        </div>
        <div class="hero-time-badge">
            <div style="font-size:10px; color:#94a3b8; text-transform:uppercase; font-weight:700;">Final Race Time</div>
            <div class="hero-time-val">{row['final_time_sec']:.2f}s</div>
        </div>
    </div>
    ''')

    def pillar_entries(*metrics):
        out = []
        for pname, entries in pillar_report["pillars"].items():
            for e in entries:
                if e["metric"] in metrics:
                    out.append(e)
        return out

    # 2. Main Telemetry Stream (Left) vs Executive Focus (Right)
    main_col, side_col = st.columns([1.8, 1])

    with main_col:
        # Chart: Speed Profile Across Quarters
        render_html('<div class="tile-card"><div class="tile-title">📈 Race Velocity Profile (m/s)</div>')
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.2), dpi=150)
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        quarters = ["0–25m", "25–50m", "50–75m", "75–100m"]
        speeds = [row["l1_first25_speed"], row["l1_second25_speed"], row["l2_first25_speed"], row["l2_second25_speed"]]

        ax.plot(quarters, speeds, marker="o", color="#38bdf8", linewidth=2, markersize=5)
        ax.fill_between(quarters, speeds, color="#38bdf8", alpha=0.12)
        ax.set_ylabel("Velocity (m/s)", fontsize=8, color="#94a3b8")
        ax.tick_params(labelsize=8, colors="#94a3b8")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color((1, 1, 1, 0.1))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        render_html('</div>')

        # Telemetry Tiles Stack
        render_tile_segment(1, "Pacing & Split Consistency",
                            pillar_entries("pacing_decay_ratio", "intra_lap1_fade_ratio", "intra_lap2_fade_ratio", "finish_vs_fresh_ratio"),
                            row, sub, positive_msg="Speed management across both laps is optimal.")

        p2 = pillar_entries("breakout_decay_ratio", "l1_breakout_pct", "l2_breakout_pct", "l1_underwater_speed", "l2_underwater_speed")
        render_tile_segment(2, "Underwater Hydrodynamics", p2, row, sub, positive_msg="Underwater efficiency off both walls is on point.")

        p3 = pillar_entries("si_retention", "l1_relative_stroke_length", "l2_relative_stroke_length", "surface_speed_decay_ratio")
        eff_rows = ""
        for lap in [1, 2]:
            key = f"l{lap}_stroke_efficiency_pct"
            label, _ = METRIC_INFO[key]
            pct, tier = stroke_eff[key]
            eff_rows += stat_row_html(label, f"{pct:.0f}%", "Reach share vs. optimal", tier)
        render_tile_segment(3, "Stroke Efficiency", p3, row, sub, eff_rows, positive_msg="Stroke mechanics remain effective under fatigue.")

        p4 = pillar_entries("l1_stroke_rate", "l2_stroke_rate")
        cadence_change = row["l2_stroke_rate"] - row["l1_stroke_rate"]
        render_tile_segment(4, "Cadence Dynamics", p4, row, sub, positive_msg=f"Cadence balanced ({cadence_change:+.1f} spm shift).")

    with side_col:
        # Simulation & Key Focus Panel
        sim, sim_reason = simulate_primary_gap(row, pillar_report, elite_features)
        if sim:
            gap_label, _ = METRIC_INFO.get(sim["gap_metric"], (sim["gap_metric"], ""))
            near = min(sim["simulated_time_conservative"], sim["simulated_time_typical"])
            far = max(sim["simulated_time_conservative"], sim["simulated_time_typical"])
            render_html(f'''
            <div class="side-panel" style="border-color: #0284c7;">
                <div class="side-panel-header" style="color:#38bdf8;">🎯 Primary Time Drop Opportunity</div>
                <div style="font-size:13px; color:#e2e8f0; margin-bottom:6px;">Target Area: <b>{gap_label}</b></div>
                <div style="font-family:'JetBrains Mono'; font-size:26px; font-weight:800; color:#38bdf8;">
                    {near:.2f}s – {far:.2f}s
                </div>
                <div style="font-size:11px; color:#6b7280; margin-top:4px;">
                    Estimated final time potential if metric reaches optimal baseline (Actual: {sim['actual_time']:.2f}s).
                </div>
            </div>
            ''')

        # Priority Action Items Matrix
        all_non_strength = []
        for pname, entries in pillar_report["pillars"].items():
            for e in entries:
                if e["severity"] != "Strength":
                    all_non_strength.append((e["severity"], e["metric"]))
        all_non_strength.sort(key=lambda x: SEVERITY_RANK[x[0]])

        actions_html = "".join(
            f'<div class="action-row">{badge_html(sev)} <span>{ACTION_TEXT.get(m, "")}</span></div>'
            for sev, m in all_non_strength
        ) or '<div class="action-row">No priority actions required — all metrics optimal.</div>'

        render_html(f'''
        <div class="side-panel">
            <div class="side-panel-header">📌 Priority Coaching Action Items</div>
            {actions_html}
        </div>
        ''')

        # Metric Tier Legend
        render_html('''
        <div class="side-panel">
            <div class="side-panel-header">ℹ️ Metric Classification Legend</div>
            <div style="font-size:11.5px; line-height:1.6; color:#9ca3af;">
                <div>• <b>Strength:</b> Performs above target benchmark.</div>
                <div>• <b>On-par:</b> Inside optimal target range.</div>
                <div>• <b>Gap:</b> Below target; clear area to address.</div>
                <div>• <b>Critical Gap:</b> Primary limit on performance.</div>
            </div>
        </div>
        ''')

else:
    st.info("Fill out the race details above or load a saved race to launch the diagnostic engine.")