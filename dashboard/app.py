"""
Lane Theory — Dashboard
Streamlit app: input a race, get a High-Performance Diagnostic report —
plain-English, segment-based, grounded entirely in real computed values
(no invented physiology constants). Every comparison is against the
optimal range for the swimmer's own gender. No ranking anywhere.

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
    compute_pillar_report, simulate_primary_gap, get_direction, PILLARS,
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
    "pacing_decay_ratio": ("Pacing Decay", "How much your second half slows down vs. your first half."),
    "intra_lap1_fade_ratio": ("Lap 1 Steadiness", "Speed retained within the first 50m."),
    "intra_lap2_fade_ratio": ("Lap 2 Steadiness", "Speed retained within the second 50m."),
    "finish_vs_fresh_ratio": ("Finishing Strength", "Final 25m pace as a share of your fastest, freshest 25m."),
    "breakout_decay_ratio": ("Breakout Consistency", "How much of your Lap 1 breakout distance you repeat on Lap 2."),
    "l1_breakout_pct": ("Lap 1 Breakout", "Distance covered underwater before surfacing off the dive."),
    "l2_breakout_pct": ("Lap 2 Breakout", "Distance covered underwater before surfacing off the turn."),
    "l1_underwater_speed": ("Lap 1 Underwater Speed", "How fast you travel underwater off the dive."),
    "l2_underwater_speed": ("Lap 2 Underwater Speed", "How fast you travel underwater off the turn."),
    "si_retention": ("Stroke Efficiency Retention", "How much of your Lap 1 stroke efficiency you keep in Lap 2."),
    "l1_relative_stroke_length": ("Lap 1 Stroke Reach", "Distance covered per stroke in Lap 1."),
    "l2_relative_stroke_length": ("Lap 2 Stroke Reach", "Distance covered per stroke in Lap 2."),
    "l1_stroke_rate": ("Lap 1 Stroke Rate", "Strokes per minute in Lap 1."),
    "l2_stroke_rate": ("Lap 2 Stroke Rate", "Strokes per minute in Lap 2."),
    "surface_speed_decay_ratio": ("Swimming Speed Retention", "Clean swimming speed (excluding underwater) kept into Lap 2."),
    "l1_stroke_efficiency_pct": ("Lap 1 Stroke Efficiency", "Your stroke reach as a share of the optimal average."),
    "l2_stroke_efficiency_pct": ("Lap 2 Stroke Efficiency", "Your stroke reach as a share of the optimal average."),
}

ACTION_TEXT = {
    "l1_underwater_speed": "Build underwater kick power off the dive.",
    "l2_underwater_speed": "Build underwater kick power off the turn wall.",
    "breakout_decay_ratio": "Extend your Lap 2 breakout distance closer to your Lap 1 breakout.",
    "l1_breakout_pct": "Extend the underwater phase off the dive (within the 15m rule).",
    "l2_breakout_pct": "Extend the underwater phase off the turn wall.",
    "pacing_decay_ratio": "Work on holding pace more evenly across both laps.",
    "si_retention": "Focus on maintaining stroke length and grip as fatigue sets in.",
    "l1_relative_stroke_length": "Focus on a longer, more complete pull in Lap 1.",
    "l2_relative_stroke_length": "Focus on a longer, more complete pull late in the race.",
    "l1_stroke_rate": "Match Lap 1 cadence to your stroke reach, rather than spinning faster.",
    "l2_stroke_rate": "Avoid over-spinning late in the race — prioritize stroke length over turnover speed.",
    "surface_speed_decay_ratio": "Work on maintaining clean swimming speed independent of turns.",
    "intra_lap1_fade_ratio": "Work on even pacing within your first 50m.",
    "intra_lap2_fade_ratio": "Work on even pacing within your second 50m.",
    "finish_vs_fresh_ratio": "Build finishing-speed endurance for the final 25m.",
}

# ---------- 5-tier classification ----------
TIER_ORDER = ["Critical", "Bad", "Medium", "Good", "Excellent"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}
TIER_BADNESS = {"Critical": 4, "Bad": 3, "Medium": 2, "Good": 1, "Excellent": 0}

TIER_STYLE = {
    "Excellent": ("#4ade80", "rgba(34,197,94,0.15)"),
    "Good": ("#60a5fa", "rgba(59,130,246,0.15)"),
    "Medium": ("#fbbf24", "rgba(245,158,11,0.15)"),
    "Bad": ("#fb923c", "rgba(249,115,22,0.15)"),
    "Critical": ("#f87171", "rgba(239,68,68,0.18)"),
}

TIER_EXPLAIN = {
    "Excellent": "Performing above the optimal benchmark — a genuine advantage.",
    "Good": "Solidly within the optimal range, close to the top performers.",
    "Medium": "Within the optimal range, but on the lower end — room to move up.",
    "Bad": "Below the optimal range — a clear area to address.",
    "Critical": "Well below the optimal range — the primary factor limiting performance.",
}

TIER_PHRASE = {
    "Excellent": "a clear strength",
    "Good": "solidly on target",
    "Medium": "within range, with room to sharpen",
    "Bad": "below target — worth addressing",
    "Critical": "your main limiter here",
}

# ---------- Per-segment visual theme (purely cosmetic — no effect on
# any computed value, tier, or action) ----------
SEGMENT_THEME = {
    1: {"icon": "🏁", "c1": "#3b82f6", "c2": "#6366f1"},   # Pacing — blue/indigo
    2: {"icon": "🌊", "c1": "#06b6d4", "c2": "#0891b2"},   # Underwater — cyan/teal
    3: {"icon": "💪", "c1": "#a855f7", "c2": "#8b5cf6"},   # Stroke Efficiency — violet
    4: {"icon": "🔄", "c1": "#f59e0b", "c2": "#f97316"},   # Cadence — amber/orange
}


# ---------- Cached loaders ----------

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


def format_race_time(seconds):
    """Show race-scale times the way swimmers/coaches actually read them:
    plain seconds under a minute (e.g. '57.53s'), mm:ss.xx at or above a
    minute (e.g. '1:02.43') — since 100m times commonly cross 60s for
    non-elite swimmers, and '62.43' isn't intuitive to read at a glance."""
    total_cs = round(seconds * 100)
    m, rem = divmod(total_cs, 6000)
    s = rem / 100
    if m > 0:
        return f"{m:02d}:{s:05.2f}"
    return f"{s:.2f}s"


def parse_time_input(text):
    """Accept a time typed either as plain seconds ('57.53') or as
    mm:ss.xx ('1:02.43') — mirrors format_race_time so what a swimmer
    types matches what they'll see displayed back. Returns None (rather
    than raising) on anything unparseable, so the caller can show a
    friendly error instead of crashing."""
    text = text.strip()
    if not text:
        return None
    try:
        if ":" in text:
            mm, ss = text.split(":", 1)
            return int(mm) * 60 + float(ss)
        return float(text)
    except ValueError:
        return None


# ---------- Severity, 5-tier refinement, readable formatting ----------

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


def refine_tier(internal_verdict, value, elite_min, elite_max, elite_mean, direction):
    if internal_verdict == "Strength":
        return "Excellent"
    if internal_verdict == "Critical Gap":
        return "Critical"
    if internal_verdict == "Gap":
        return "Bad"
    if direction == "higher":
        midpoint = (elite_min + elite_mean) / 2
        return "Good" if value >= midpoint else "Medium"
    else:
        midpoint = (elite_mean + elite_max) / 2
        return "Good" if value <= midpoint else "Medium"


def format_metric(entry, row, sub):
    m, v, mean = entry["metric"], entry["value"], entry["elite_mean"]

    if m == "pacing_decay_ratio":
        return f"{(v - 1) * 100:+.1f}%", f"Optimal average: {(mean - 1) * 100:+.1f}%"

    if m in ("finish_vs_fresh_ratio", "breakout_decay_ratio", "si_retention",
             "surface_speed_decay_ratio", "intra_lap1_fade_ratio", "intra_lap2_fade_ratio"):
        return f"{v * 100:.0f}%", f"Optimal average: {mean * 100:.0f}%"

    if m in ("l1_breakout_pct", "l2_breakout_pct"):
        lap = m[1]
        raw_val = row[f"l{lap}_breakout_distance"]
        raw_mean = sub[f"l{lap}_breakout_distance"].mean()
        return f"{raw_val:.1f} m", f"Optimal average: {raw_mean:.1f} m"

    if m in ("l1_underwater_speed", "l2_underwater_speed"):
        return f"{v:.2f} m/s", f"Optimal average: {mean:.2f} m/s"

    if m in ("l1_stroke_rate", "l2_stroke_rate"):
        return f"{v:.0f} spm", f"Optimal average: {mean:.0f} spm"

    if m in ("l1_relative_stroke_length", "l2_relative_stroke_length"):
        lap = m[1]
        raw_val = row[f"l{lap}_stroke_length"]
        raw_mean = sub[f"l{lap}_stroke_length"].mean()
        return f"{raw_val:.2f} m", f"Optimal average: {raw_mean:.2f} m"

    return f"{v:.3f}", f"Optimal average: {mean:.3f}"


def compute_stroke_efficiency(row, elite_df, entry_lookup):
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]
    result = {}
    for lap in [1, 2]:
        col = f"l{lap}_relative_stroke_length"
        mean = sub[col].mean()
        pct = (row[col] / mean) * 100
        result[f"l{lap}_stroke_efficiency_pct"] = (pct, entry_lookup[col]["tier"])
    return result


# ---------- CSS ----------

def inject_css():
    st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1.5rem; max-width: 820px;}
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #0b1120;}

.hero-card {background: linear-gradient(135deg, #111827 0%, #0b1120 100%), repeating-linear-gradient(115deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 2px, transparent 2px, transparent 46px); border: 1px solid #1f2937; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; box-shadow: 0 8px 28px rgba(59,130,246,0.12); position: relative; overflow: hidden;}
.hero-card::after {content: ""; position: absolute; top: -40%; right: -10%; width: 220px; height: 220px; background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%); pointer-events: none;}
.hero-label {font-size: 11px; letter-spacing: 1.4px; color: #64748b; text-transform: uppercase; font-weight: 700;}
.hero-name {font-size: 28px; font-weight: 800; color: #f8fafc; margin: 2px 0 12px 0;}
.hero-meta {display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: #94a3b8;}
.time-box {border: 1px solid #1f2937; border-radius: 12px; padding: 10px 18px; text-align: center; background: #0b1120;}
.time-label {font-size: 10.5px; letter-spacing: 1px; color: #64748b; text-transform: uppercase; font-weight: 700;}
.time-num {font-size: 30px; font-weight: 800; color: #f8fafc;}

.seg-card {background: #111827; border-radius: 14px; padding: 14px 20px; margin-bottom: 12px; border: 1px solid #1f2937; border-top: 3px solid var(--seg-c1, #3b82f6); transition: transform 0.15s ease, box-shadow 0.15s ease;}
.seg-card:hover {transform: translateY(-2px); box-shadow: 0 8px 22px rgba(0,0,0,0.35);}
.seg-header {display: flex; align-items: center; gap: 10px; margin-bottom: 10px;}
.seg-num {background: linear-gradient(135deg, var(--seg-c1, #3b82f6), var(--seg-c2, #6366f1)); color: white; width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; flex-shrink: 0; box-shadow: 0 3px 10px rgba(0,0,0,0.3);}
.seg-title {font-weight: 700; font-size: 13px; letter-spacing: 0.6px; color: #f1f5f9; text-transform: uppercase;}

.pos-track {position: relative; height: 5px; border-radius: 3px; margin-top: 7px; margin-bottom: 2px;}
.pos-marker {position: absolute; top: -2.5px; width: 10px; height: 10px; border-radius: 50%; background: #f8fafc; box-shadow: 0 0 6px rgba(0,0,0,0.5);}

.stats-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0 20px;}
.stat-row {display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1f2937;}
.stat-row:last-child {border-bottom: none;}
.stat-label {font-size: 12.5px; color: #cbd5e1; font-weight: 500;}
.stat-opt {font-size: 10.5px; color: #64748b; margin-top: 1px;}
.stat-value {font-size: 15px; font-weight: 700; color: #f8fafc; text-align: right;}

.badge {display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 9.5px; font-weight: 700; text-transform: uppercase; margin-left: 6px; letter-spacing: 0.3px;}

.diag-box {background: #0b1120; border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 12.5px; color: #cbd5e1; line-height: 1.5;}
.diag-title {font-weight: 700; font-size: 11px; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.5px; margin: 8px 0 3px 0;}
.diag-title:first-child {margin-top: 0;}
.diag-box ul {margin: 2px 0 0 0; padding-left: 16px;}
.diag-box li {margin-bottom: 2px;}
.maintain-line {color: #94a3b8; font-size: 12px;}

.opportunity-card {background: linear-gradient(135deg, rgba(59,130,246,0.14) 0%, rgba(99,102,241,0.06) 100%); border: 1px solid rgba(59,130,246,0.35); border-radius: 14px; padding: 20px 24px; margin: 8px 0 14px 0;}
.opportunity-range {font-size: 30px; font-weight: 800; color: #60a5fa; margin: 8px 0 4px 0; animation: pulse-glow 2.4s ease-in-out infinite;}
@keyframes pulse-glow {
    0%, 100% {text-shadow: 0 0 6px rgba(96,165,250,0.25);}
    50% {text-shadow: 0 0 22px rgba(96,165,250,0.75);}
}

.bottom-card {background: #111827; border-radius: 14px; padding: 16px 20px; border: 1px solid #1f2937; height: 100%;}
.bottom-title {font-weight: 700; font-size: 12px; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;}
.action-item {padding: 6px 0; font-size: 13px; color: #cbd5e1; border-bottom: 1px solid #1f2937;}
.action-item:last-child {border-bottom: none;}

.calc-box {background: #0b1120; border: 1px dashed #334155; border-radius: 10px; padding: 10px 14px; margin-top: 4px; font-size: 13px; color: #94a3b8;}
.calc-box b {color: #f8fafc;}
</style>
    """, unsafe_allow_html=True)


def badge(tier):
    fg, bg = TIER_STYLE[tier]
    return f"<span class='badge' style='color:{fg}; background:{bg};'>{tier}</span>"


def stat_row_formatted(label, value_str, caption_str, tier):
    return (f'<div class="stat-row"><div><div class="stat-label">{label}</div>'
            f'<div class="stat-opt">{caption_str}</div></div>'
            f'<div class="stat-value">{value_str}{badge(tier)}</div></div>')


def diag_icon(tiers):
    if "Critical" in tiers:
        return "🛑"
    if "Bad" in tiers:
        return "⚠️"
    return "✅"


def render_segment(num, title, entries, row, sub, extra_stats_html=""):
    theme = SEGMENT_THEME[num]

    stats_html = ""
    maintain_items, improve_items = [], []
    for e in entries:
        label, _ = METRIC_INFO.get(e["metric"], (e["metric"], ""))
        value_str, caption_str = format_metric(e, row, sub)
        stats_html += stat_row_formatted(label, value_str, caption_str, e["tier"])

        if e["tier"] in ("Excellent", "Good"):
            maintain_items.append(label)
        else:
            action = ACTION_TEXT.get(e["metric"], "")
            if action:
                improve_items.append(f"<li>{action}</li>")

    stats_html += extra_stats_html

    diag_html = ""
    if maintain_items:
        diag_html += f'<div class="diag-title">✅ Maintain</div><div class="maintain-line">{", ".join(maintain_items)}</div>'
    if improve_items:
        diag_html += f'<div class="diag-title">🔧 To Improve</div><ul>{"".join(improve_items)}</ul>'
    if not diag_html:
        diag_html = '<div class="maintain-line">Nothing flagged — all metrics at or above target.</div>'

    render_html(f'''
    <div class="seg-card" style="--seg-c1:{theme['c1']}; --seg-c2:{theme['c2']};">
        <div class="seg-header"><div class="seg-num">{theme['icon']}</div><div class="seg-title">{title}</div></div>
        <div class="stats-grid">{stats_html}</div>
        <div class="diag-box">{diag_html}</div>
    </div>
    ''')


# ---------- App ----------

st.set_page_config(page_title="Lane Theory", layout="centered")
inject_css()

elite_features = load_elite_features()
models = load_models()
shap_values_dict = load_shap_values()
individuals_raw = load_individuals_raw()

st.title("🏊 Lane Theory")
st.caption("A personal race-strategy diagnostic — not a leaderboard. Every number is compared "
           "against the optimal range for your gender, drawn from a benchmark of fast 100m Freestyle swims.")

tab_new, tab_existing = st.tabs(["➕ New race entry", "📂 View a saved race"])

with tab_new:
    with st.container(border=True):
        st.markdown("**🧍 Swimmer Details**")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name", key="in_name")
            gender = st.selectbox("Gender", ["male", "female"], key="in_gender")
        with col2:
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0, key="in_height")
            dob = st.date_input("Date of birth", value=date(2005, 1, 1), key="in_dob")
        with col3:
            race_date_input = st.date_input("Race date", value=date.today(), key="in_race_date")
            pb_50m_seconds = st.number_input("50m PB (s)", min_value=18.0, max_value=60.0, value=26.5, key="in_pb50")

    with st.container(border=True):
        st.markdown("**⏱️ Race Result**")
        col4a, col4b = st.columns(2)
        final_time_text = col4a.text_input("Final time (e.g. 57.53 or 1:02.43)", value="57.53", key="in_final_text")
        l1_reaction_time = col4b.number_input("Reaction time (s)", min_value=0.3, max_value=1.5, value=0.67, key="in_rt")
        final_time_sec = parse_time_input(final_time_text)
        if final_time_sec is None:
            st.error("Enter a valid time, like 57.53 or 1:02.43.")
        else:
            render_html(f'<div class="calc-box">Final time: <b>{format_race_time(final_time_sec)}</b></div>')

    with st.container(border=True):
        st.markdown("**1️⃣ Lap 1 (0–50m)**")
        c1, c2 = st.columns(2)
        l1_breakout_distance = c1.number_input("Breakout distance (m)", 0.0, 15.0, 12.0, key="l1bd")
        l1_breakout_time = c2.number_input("Breakout time (s)", 0.5, 10.0, 4.6, key="l1bt")
        c3, c4 = st.columns(2)
        l1_stroke_count = c3.number_input("Stroke count", 1, 60, 37, key="l1sc")
        l1_total_time = c4.number_input("Lap 1 total time (s)", 10.0, 60.0, 27.81, key="l1tt")
        l1_split_25m = st.number_input("Cumulative time @ 25m (s)", 8.0, 40.0, 12.3, key="l1s25")

    with st.container(border=True):
        st.markdown("**2️⃣ Lap 2 (50–100m)**")
        c5, c6 = st.columns(2)
        l2_breakout_distance = c5.number_input("Breakout distance (m)", 0.0, 15.0, 6.5, key="l2bd")
        l2_breakout_time = c6.number_input("Breakout time (s)", 0.5, 10.0, 2.7, key="l2bt")
        l2_stroke_count = st.number_input("Stroke count", 1, 60, 45, key="l2sc")
        l2_split_25m = st.number_input("Time @ 75m mark (lap-relative, s)", 8.0, 40.0, 14.2, key="l2s25")

        if final_time_sec is None:
            l2_total_time = None
            st.warning("Enter a valid final time above to calculate Lap 2 total time.")
        else:
            l2_total_time = final_time_sec - l1_total_time
            if l2_total_time <= 0:
                st.warning("Lap 1 total time can't be greater than or equal to the final time — check those two values.")
            else:
                render_html(f'<div class="calc-box">Lap 2 total time (auto-calculated: final time − Lap 1 time): '
                            f'<b>{format_race_time(l2_total_time)}</b></div>')

    run_new = st.button("🏁 Run Diagnosis", key="run_new", width="stretch")

    if run_new and final_time_sec is None:
        st.error("Fix the final time field above before running the diagnosis.")
    elif run_new and (l2_total_time is None or l2_total_time <= 0):
        st.error("Lap 1 and final time don't add up — fix those before running the diagnosis.")
    elif run_new:
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
    if individuals_raw.empty:
        st.info("No races saved yet — add one in the 'New race entry' tab.")
    else:
        df = individuals_raw.copy()
        df["occurrence"] = (df.groupby("name").cumcount() + 1).astype(int)
        df["display_label"] = df.apply(
            lambda r: f"{r['name']} ({format_race_time(r['final_time_sec'])}) #{int(r['occurrence'])}", axis=1
        )
        label_to_race_id = dict(zip(df["display_label"], df["race_id"]))

        chosen_label = st.selectbox("Choose a swimmer / race", df["display_label"].tolist())
        if st.button("🏁 Run Diagnosis", key="run_existing", width="stretch"):
            chosen_race_id = label_to_race_id[chosen_label]
            selected_raw = individuals_raw[individuals_raw["race_id"] == chosen_race_id]
            row_df = engineer_features(selected_raw)
            st.session_state["diagnosis_row"] = row_df.iloc[0]


# =======================================================================
# DIAGNOSIS REPORT
# =======================================================================

if "diagnosis_row" in st.session_state:
    row = st.session_state["diagnosis_row"]
    gender = row["gender"]
    sub = elite_features[elite_features["gender"] == gender]

    pillar_report = compute_pillar_report(row, elite_features)

    for pillar, entries in pillar_report["pillars"].items():
        for e in entries:
            e["severity"] = compute_severity(e["metric"], e["value"], e["elite_min"],
                                              e["elite_max"], e["elite_mean"], e["verdict"], sub)
            direction = get_direction(e["metric"], sub)
            e["tier"] = refine_tier(e["severity"], e["value"], e["elite_min"], e["elite_max"], e["elite_mean"], direction)

    entry_lookup = {e["metric"]: e for entries in pillar_report["pillars"].values() for e in entries}
    stroke_eff = compute_stroke_efficiency(row, elite_features, entry_lookup)
    age_years = row.get("age_at_race", None)

    # ---------- Possible Lap 1 / Lap 2 data-swap warning ----------
    swap_indicators = []
    if row["pacing_decay_ratio"] < 1:
        swap_indicators.append("pacing")
    if row["si_retention"] > 1:
        swap_indicators.append("stroke efficiency")
    if row["surface_speed_decay_ratio"] > 1:
        swap_indicators.append("surface speed")
    if row["l2_relative_stroke_length"] > row["l1_relative_stroke_length"]:
        swap_indicators.append("stroke reach")
    if len(swap_indicators) >= 3:
        st.warning(
            f"Every one of these metrics suggests Lap 2 outperformed Lap 1: "
            f"{', '.join(swap_indicators)}. This can happen with a genuine negative-split "
            f"swim, but a clean sweep across every dimension at once is also a classic sign "
            f"that Lap 1 and Lap 2 data got swapped when entering this race. Worth "
            f"double-checking your entries if this wasn't a deliberate negative split."
        )

    render_html(f'''
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:14px;">
            <div>
                <div class="hero-label">High-Performance Diagnostic</div>
                <div class="hero-name">{row['name']}</div>
                <div class="hero-meta">
                    <span>👤 {gender.capitalize()}</span>
                    <span>📏 {row['height_cm']:.0f} cm</span>
                    <span>🎂 {age_years:.0f} yrs</span>
                    <span>🏊 100m Freestyle (LCM)</span>
                </div>
            </div>
            <div class="time-box">
                <div class="time-label">Actual Final Time</div>
                <div class="time-num">{format_race_time(row['final_time_sec'])}</div>
            </div>
        </div>
    </div>
    ''')

    def pillar_entries(*metrics):
        return [e for pname, entries in pillar_report["pillars"].items() for e in entries if e["metric"] in metrics]

    render_segment(1, "Pacing & Split Consistency",
                    pillar_entries("pacing_decay_ratio", "intra_lap1_fade_ratio", "intra_lap2_fade_ratio", "finish_vs_fresh_ratio"),
                    row, sub)

    render_segment(2, "Underwater Hydrodynamics",
                    pillar_entries("breakout_decay_ratio", "l1_breakout_pct", "l2_breakout_pct", "l1_underwater_speed", "l2_underwater_speed"),
                    row, sub)

    p3 = pillar_entries("si_retention", "l1_relative_stroke_length", "l2_relative_stroke_length", "surface_speed_decay_ratio")
    eff_rows = ""
    for lap in [1, 2]:
        key = f"l{lap}_stroke_efficiency_pct"
        label, _ = METRIC_INFO[key]
        pct, tier = stroke_eff[key]
        eff_rows += stat_row_formatted(label, f"{pct:.0f}%", "Share of the optimal average reach", tier)
    render_segment(3, "Stroke Efficiency", p3, row, sub, extra_stats_html=eff_rows)

    render_segment(4, "Cadence Dynamics", pillar_entries("l1_stroke_rate", "l2_stroke_rate"), row, sub)

    st.markdown("**🌊 Speed profile across the race**")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(3.4, 1.3), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    quarters = ["0–25m", "25–50m", "50–75m", "75–100m"]
    speeds = [row["l1_first25_speed"], row["l1_second25_speed"], row["l2_first25_speed"], row["l2_second25_speed"]]
    x_pos = range(len(quarters))
    ax.plot(x_pos, speeds, marker="o", markersize=5, color="#38bdf8", linewidth=2, zorder=3,
            markerfacecolor="#0b1120", markeredgecolor="#38bdf8", markeredgewidth=1.6)
    ax.fill_between(x_pos, speeds, min(speeds) * 0.9, color="#38bdf8", alpha=0.15, zorder=1)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(quarters)
    ax.set_ylim(bottom=min(speeds) * 0.9)
    ax.set_ylabel("m/s", fontsize=7, color="#cbd5e1")
    ax.tick_params(labelsize=6.5, colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    col_chart, _ = st.columns([1, 1])
    with col_chart:
        st.pyplot(fig, width="stretch")

    st.write("")

    # ---------- Unified 5-tier "most limiting pillar" + full action list ----------
    pillar_badness = {}
    all_needs_work = []
    for pname, entries in pillar_report["pillars"].items():
        pillar_badness[pname] = sum(TIER_BADNESS[e["tier"]] for e in entries)
        for e in entries:
            if e["tier"] in ("Medium", "Bad", "Critical"):
                all_needs_work.append((e["tier"], e["metric"]))
    all_needs_work.sort(key=lambda x: TIER_RANK[x[0]])

    max_badness = max(pillar_badness.values()) if pillar_badness else 0
    biggest_gap_pillars = [p for p, s in pillar_badness.items() if s == max_badness] if max_badness > 0 else None

    sim, sim_reason = simulate_primary_gap(row, pillar_report, elite_features)
    if sim:
        gap_label, _ = METRIC_INFO.get(sim["gap_metric"], (sim["gap_metric"], ""))
        near = min(sim["simulated_time_conservative"], sim["simulated_time_typical"])
        far = max(sim["simulated_time_conservative"], sim["simulated_time_typical"])
        extra_line = ""
        if len(all_needs_work) > 1:
            extra_line = (" This reflects fixing only that one factor — see Priority Coaching Action "
                          "Items below for everything else with room to improve.")
        render_html(f'''
        <div class="opportunity-card">
            <div class="bottom-title">🎯 Primary Time-Drop Opportunity</div>
            <div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">
                Prediction given if swimmer fixes their gap to a realistic point within the optimal range:
            </div>
            <div class="opportunity-range">{format_race_time(near)} – {format_race_time(far)}</div>
            <div style="font-size:11.5px; color:#64748b;">against your actual {format_race_time(sim['actual_time'])}.
                A physics-based simulation, not a guarantee.{extra_line}</div>
        </div>
        ''')

    if biggest_gap_pillars:
        takeaway = (f"{row['name']}'s performance is most limited by "
                    f"{', '.join(biggest_gap_pillars)}. Focused training here offers the biggest opportunity to drop time.")
    else:
        takeaway = f"{row['name']} is performing within or above the optimal range across every metric measured — a well-rounded, strong performance."

    actions_html = "".join(
        f'<div class="action-item">{badge(tier)} {ACTION_TEXT.get(m, "")}</div>'
        for tier, m in all_needs_work
    ) or '<div class="action-item">No priority actions — every metric is at or above the optimal average.</div>'

    col_a, col_b = st.columns(2)
    with col_a:
        render_html(f'<div class="bottom-card"><div class="bottom-title">🎯 Key Takeaway</div>'
                    f'<div style="font-size:13px; color:#cbd5e1; line-height:1.55;">{takeaway}</div></div>')
    with col_b:
        render_html(f'<div class="bottom-card"><div class="bottom-title">📌 Priority Coaching Action Items</div>{actions_html}</div>')

    with st.expander("ℹ️ Metric Classification Legend"):
        for tier in TIER_ORDER[::-1]:
            st.markdown(f"{badge(tier)} &nbsp; {TIER_EXPLAIN[tier]}", unsafe_allow_html=True)
        st.caption("The 'optimal range' is drawn from a benchmark sample of 20 male and 20 female "
                   "fast 100m Freestyle performances — a reference range, not a ranking.")

else:
    st.info("Enter a new race or load a saved one above to see your diagnostic report.")

