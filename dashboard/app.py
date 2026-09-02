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
    "pacing_decay_ratio": ("Pacing Decay Ratio", "How much your second half slows down vs. your first half."),
    "intra_lap1_fade_ratio": ("Lap 1 Steadiness", "How well you hold your speed within the first 50m."),
    "intra_lap2_fade_ratio": ("Lap 2 Steadiness", "How well you hold your speed within the second 50m."),
    "finish_vs_fresh_ratio": ("Finishing Strength", "Final 25m speed compared to your fastest, freshest 25m."),
    "breakout_decay_ratio": ("Breakout Retention", "Underwater distance kept on your second wall push vs. your first."),
    "l1_breakout_pct": ("Lap 1 Underwater Share", "Share of your first lap spent underwater off the dive."),
    "l2_breakout_pct": ("Lap 2 Underwater Share", "Share of your second lap spent underwater off the turn."),
    "l1_underwater_speed": ("Lap 1 Underwater Speed", "How fast you travel underwater off the dive."),
    "l2_underwater_speed": ("Lap 2 Underwater Speed", "How fast you travel underwater off the turn."),
    "si_retention": ("Stroke Efficiency Retention", "How much of your Lap 1 stroke efficiency you keep in Lap 2."),
    "l1_relative_stroke_length": ("Lap 1 Stroke Reach", "Distance covered per stroke in Lap 1, scaled to your height."),
    "l2_relative_stroke_length": ("Lap 2 Stroke Reach", "Distance covered per stroke in Lap 2, scaled to your height."),
    "l1_stroke_rate": ("Lap 1 Stroke Rate", "Strokes per minute in Lap 1."),
    "l2_stroke_rate": ("Lap 2 Stroke Rate", "Strokes per minute in Lap 2."),
    "surface_speed_decay_ratio": ("Swimming Speed Retention", "Clean swimming speed (excluding underwater) kept into Lap 2."),
    "l1_stroke_efficiency_pct": ("Lap 1 Stroke Efficiency", "Your stroke reach scored against the optimal range."),
    "l2_stroke_efficiency_pct": ("Lap 2 Stroke Efficiency", "Your stroke reach scored against the optimal range."),
}

ACTION_TEXT = {
    "l1_underwater_speed": "Build underwater kick power off the dive.",
    "l2_underwater_speed": "Build underwater kick power off the turn wall.",
    "breakout_decay_ratio": "Extend your Lap 2 breakout distance closer to your Lap 1 breakout.",
    "l1_breakout_pct": "Consider a longer underwater phase off the dive (within the legal limit).",
    "l2_breakout_pct": "Consider a longer underwater phase off the turn wall.",
    "pacing_decay_ratio": "Work on holding pace more evenly across both laps.",
    "si_retention": "Focus on maintaining stroke length and grip as fatigue sets in.",
    "l1_relative_stroke_length": "Focus on a longer, more complete pull in Lap 1.",
    "l2_relative_stroke_length": "Focus on a longer, more complete pull late in the race.",
    "l1_stroke_rate": "Review Lap 1 cadence relative to your stroke length.",
    "l2_stroke_rate": "Avoid over-spinning late in the race — prioritize stroke length over turnover speed.",
    "surface_speed_decay_ratio": "Work on maintaining clean swimming speed independent of turns.",
    "intra_lap1_fade_ratio": "Work on even pacing within your first 50m.",
    "intra_lap2_fade_ratio": "Work on even pacing within your second 50m.",
    "finish_vs_fresh_ratio": "Build finishing-speed endurance for the final 25m.",
    "l1_stroke_efficiency_pct": "Work on stroke reach and power together in Lap 1.",
    "l2_stroke_efficiency_pct": "Work on stroke reach and power together in Lap 2.",
}

VERDICT_STYLE = {
    "Strength": ("#4ade80", "rgba(34,197,94,0.15)", "Strength"),
    "On-par": ("#60a5fa", "rgba(59,130,246,0.15)", "On-par"),
    "Gap": ("#fb923c", "rgba(249,115,22,0.15)", "Gap"),
    "Critical Gap": ("#f87171", "rgba(239,68,68,0.18)", "Critical Gap"),
}

VERDICT_EXPLAIN = {
    "Strength": "At or above the optimal range — a genuine advantage.",
    "On-par": "Inside the optimal range, with room to move closer to the top of it.",
    "Gap": "Outside the optimal range — a real opportunity for training focus.",
    "Critical Gap": "Well outside the optimal range — likely the single biggest opportunity to drop time.",
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


# ---------- HTML rendering helper (fixes indentation-as-code-block bug) ----------

def render_html(html):
    """Collapse all newlines/leading whitespace before handing to
    st.markdown — Streamlit's Markdown parser treats lines indented 4+
    spaces as a code block, which corrupted multi-line f-string HTML
    templates. Flattening to one line avoids that entirely."""
    flat = re.sub(r"\n\s*", "", html)
    st.markdown(flat, unsafe_allow_html=True)


# ---------- Severity (4-tier) + stroke efficiency % ----------

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


def compute_stroke_efficiency(row, elite_df, pillar_report):
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]
    verdict_lookup = {e["metric"]: e["severity"] for entries in pillar_report["pillars"].values() for e in entries}
    result = {}
    for lap in [1, 2]:
        rsl_col = f"l{lap}_relative_stroke_length"
        lo, hi = sub[rsl_col].min(), sub[rsl_col].max()
        pct = np.clip((row[rsl_col] - lo) / (hi - lo), 0, 1) * 100
        result[f"l{lap}_stroke_efficiency_pct"] = (pct, verdict_lookup.get(rsl_col, "On-par"))
    return result


# ---------- CSS ----------

def inject_css():
    st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 880px;}
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #0b1120;}

.hero-card {background: linear-gradient(135deg, #111827 0%, #0b1120 100%); border: 1px solid #1f2937; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(59,130,246,0.08);}
.hero-label {font-size: 11px; letter-spacing: 1.4px; color: #64748b; text-transform: uppercase; font-weight: 700;}
.hero-name {font-size: 28px; font-weight: 800; color: #f8fafc; margin: 2px 0 12px 0;}
.hero-meta {display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: #94a3b8;}
.time-box {border: 1px solid #1f2937; border-radius: 12px; padding: 10px 18px; text-align: center; background: #0b1120;}
.time-label {font-size: 10.5px; letter-spacing: 1px; color: #64748b; text-transform: uppercase; font-weight: 700;}
.time-num {font-size: 30px; font-weight: 800; color: #f8fafc;}

.seg-card {background: #111827; border-radius: 14px; padding: 18px 22px; margin-bottom: 4px; border: 1px solid #1f2937;}
.seg-header {display: flex; align-items: center; gap: 10px; margin-bottom: 4px;}
.seg-num {background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; width: 24px; height: 24px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 12px; flex-shrink: 0;}
.seg-title {font-weight: 700; font-size: 13px; letter-spacing: 0.6px; color: #f1f5f9; text-transform: uppercase;}

.stat-row {display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #1f2937;}
.stat-row:last-child {border-bottom: none;}
.stat-label {font-size: 12.5px; color: #cbd5e1; font-weight: 500;}
.stat-opt {font-size: 10.5px; color: #64748b; margin-top: 1px;}
.stat-value {font-size: 15px; font-weight: 700; color: #f8fafc; text-align: right;}

.badge {display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 9.5px; font-weight: 700; text-transform: uppercase; margin-left: 6px; letter-spacing: 0.3px;}

.diag-box {background: #0b1120; border-radius: 10px; padding: 14px; border-left: 3px solid #3b82f6; font-size: 12.5px; color: #cbd5e1; line-height: 1.55;}
.diag-title {font-weight: 700; font-size: 11.5px; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;}

.opportunity-card {background: linear-gradient(135deg, rgba(59,130,246,0.14) 0%, rgba(99,102,241,0.06) 100%); border: 1px solid rgba(59,130,246,0.35); border-radius: 14px; padding: 20px 24px; margin: 8px 0 14px 0;}
.opportunity-range {font-size: 28px; font-weight: 800; color: #60a5fa; margin: 8px 0 4px 0;}

.bottom-card {background: #111827; border-radius: 14px; padding: 16px 20px; border: 1px solid #1f2937; height: 100%;}
.bottom-title {font-weight: 700; font-size: 12px; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;}
.action-item {padding: 5px 0; font-size: 13px; color: #cbd5e1; border-bottom: 1px solid #1f2937;}
.action-item:last-child {border-bottom: none;}
</style>
    """, unsafe_allow_html=True)


def badge(verdict):
    fg, bg, label = VERDICT_STYLE[verdict]
    return f"<span class='badge' style='color:{fg}; background:{bg};'>{label}</span>"


def stat_row(metric, value, elite_mean, verdict, unit=""):
    label, _ = METRIC_INFO.get(metric, (metric, ""))
    return (f'<div class="stat-row"><div><div class="stat-label">{label}</div>'
            f'<div class="stat-opt">Optimal mean: {elite_mean:.3f}{unit}</div></div>'
            f'<div class="stat-value">{value:.3f}{unit}{badge(verdict)}</div></div>')


def diag_icon(entries_severity):
    if any(v == "Critical Gap" for v in entries_severity):
        return "🛑", "#f87171"
    if any(v == "Gap" for v in entries_severity):
        return "⚠️", "#fb923c"
    return "✅", "#3b82f6"


def render_segment(num, title, entries, extra_stats_html, positive_msg, action_extra=""):
    severities = [e["severity"] for e in entries]
    icon, color = diag_icon(severities)
    stats_html = "".join(stat_row(e["metric"], e["value"], e["elite_mean"], e["severity"]) for e in entries)
    stats_html += extra_stats_html

    gaps = [e["metric"] for e in entries if e["severity"] in ("Gap", "Critical Gap")]
    if gaps:
        diag_text = " ".join(ACTION_TEXT.get(g, "") for g in gaps) + (" " + action_extra if action_extra else "")
    else:
        diag_text = positive_msg

    render_html(f'<div class="seg-card"><div class="seg-header">'
                f'<div class="seg-num">{num}</div><div class="seg-title">{title}</div></div></div>')

    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        render_html(f'<div class="seg-card" style="margin-top:-18px; border-top:none; border-radius:0 0 14px 14px;">{stats_html}</div>')
    with col_right:
        render_html(f'<div class="diag-box" style="border-left-color:{color}; margin-top:8px;">'
                    f'<div class="diag-title">{icon} Diagnostic</div>{diag_text}</div>')

    st.write("")


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
    with st.form("new_race_form"):
        st.markdown("**Swimmer details**")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name")
            gender = st.selectbox("Gender", ["male", "female"])
        with col2:
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0)
            dob = st.date_input("Date of birth", value=date(2005, 1, 1))
        with col3:
            race_date_input = st.date_input("Race date", value=date.today())
            pb_50m_seconds = st.number_input("50m PB (s)", min_value=18.0, max_value=60.0, value=26.5)

        st.markdown("**Race result**")
        col4, col5 = st.columns(2)
        final_time_sec = col4.number_input("Final time (s)", min_value=40.0, max_value=180.0, value=57.53)
        l1_reaction_time = col5.number_input("Reaction time (s)", min_value=0.3, max_value=1.5, value=0.67)

        st.markdown("**Lap 1 (0–50m)**")
        c1, c2, c3, c4 = st.columns(4)
        l1_breakout_distance = c1.number_input("Breakout dist. (m)", 0.0, 15.0, 12.0, key="l1bd")
        l1_breakout_time = c2.number_input("Breakout time (s)", 0.5, 10.0, 4.6, key="l1bt")
        l1_stroke_count = c3.number_input("Stroke count", 1, 60, 37, key="l1sc")
        l1_total_time = c4.number_input("Lap total time (s)", 10.0, 60.0, 27.81, key="l1tt")
        l1_split_25m = st.number_input("Cumulative time @ 25m (s)", 8.0, 40.0, 12.3, key="l1s25")

        st.markdown("**Lap 2 (50–100m)**")
        c5, c6, c7, c8 = st.columns(4)
        l2_breakout_distance = c5.number_input("Breakout dist. (m)", 0.0, 15.0, 6.5, key="l2bd")
        l2_breakout_time = c6.number_input("Breakout time (s)", 0.5, 10.0, 2.7, key="l2bt")
        l2_stroke_count = c7.number_input("Stroke count", 1, 60, 45, key="l2sc")
        l2_total_time = c8.number_input("Lap total time (s)", 10.0, 60.0, 29.72, key="l2tt")
        l2_split_25m = st.number_input("Time @ 75m mark (lap-relative, s)", 8.0, 40.0, 14.2, key="l2s25")

        submitted = st.form_submit_button("🏁 Run Diagnosis", use_container_width=True)

    if submitted:
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
        df["occurrence"] = df.groupby("name").cumcount() + 1
        df["display_label"] = df.apply(
            lambda r: f"{r['name']} ({r['final_time_sec']:.2f}s) #{r['occurrence']}", axis=1
        )
        label_to_race_id = dict(zip(df["display_label"], df["race_id"]))

        chosen_label = st.selectbox("Choose a swimmer / race", df["display_label"].tolist())
        if st.button("🏁 Run Diagnosis", key="run_existing", use_container_width=True):
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
    biggest_gap_pillars = find_biggest_gap_pillar(pillar_report)

    for pillar, entries in pillar_report["pillars"].items():
        for e in entries:
            e["severity"] = compute_severity(e["metric"], e["value"], e["elite_min"],
                                              e["elite_max"], e["elite_mean"], e["verdict"], sub)

    stroke_eff = compute_stroke_efficiency(row, elite_features, pillar_report)
    age_years = row.get("age_at_race", None)

    # ---------- Hero ----------
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
                <div class="time-num">{row['final_time_sec']:.2f}s</div>
            </div>
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

    render_segment(1, "Pacing & Split Consistency",
                    pillar_entries("pacing_decay_ratio", "intra_lap1_fade_ratio", "intra_lap2_fade_ratio", "finish_vs_fresh_ratio"),
                    "", "Your pacing and within-lap consistency are strong — you hold your speed well across the race.")

    p2 = pillar_entries("breakout_decay_ratio", "l1_breakout_pct", "l2_breakout_pct", "l1_underwater_speed", "l2_underwater_speed")
    extra_note = ""
    l2_gap_entry = next((e for e in p2 if e["metric"] == "l2_breakout_pct" and e["severity"] in ("Gap", "Critical Gap")), None)
    if l2_gap_entry:
        meters_left = sub["l2_breakout_distance"].mean() - row["l2_breakout_distance"]
        if meters_left > 0:
            extra_note = f"Swimmers in the optimal range push off roughly {meters_left:.1f}m further underwater on the second wall than you currently do."
    render_segment(2, "Underwater Hydrodynamics", p2, "",
                    "Your underwater phases are strong on both walls.", action_extra=extra_note)

    p3 = pillar_entries("si_retention", "l1_relative_stroke_length", "l2_relative_stroke_length", "surface_speed_decay_ratio")
    eff_rows = ""
    for lap in [1, 2]:
        key = f"l{lap}_stroke_efficiency_pct"
        label, _ = METRIC_INFO[key]
        pct, tier = stroke_eff[key]
        eff_rows += (f'<div class="stat-row"><div><div class="stat-label">{label}</div>'
                     f'<div class="stat-opt">Position within the optimal stroke-reach range</div></div>'
                     f'<div class="stat-value">{pct:.0f}%{badge(tier)}</div></div>')
    render_segment(3, "Stroke Efficiency", p3, eff_rows,
                    "Your stroke mechanics hold up well under fatigue — strong technique retention.")

    p4 = pillar_entries("l1_stroke_rate", "l2_stroke_rate")
    cadence_change = row["l2_stroke_rate"] - row["l1_stroke_rate"]
    render_segment(4, "Cadence Dynamics", p4, "",
                    f"Your cadence is well controlled ({cadence_change:+.1f} spm change into Lap 2).",
                    action_extra=f"Your stroke rate changed by {cadence_change:+.1f} spm from Lap 1 to Lap 2.")

    st.markdown("**Speed profile across the race**")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(3.6, 1.6), dpi=140)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    quarters = ["0–25m", "25–50m", "50–75m", "75–100m"]
    speeds = [row["l1_first25_speed"], row["l1_second25_speed"], row["l2_first25_speed"], row["l2_second25_speed"]]
    ax.plot(quarters, speeds, marker="o", markersize=4, color="#60a5fa", linewidth=1.6)
    ax.set_ylabel("m/s", fontsize=7, color="#cbd5e1")
    ax.tick_params(labelsize=6.5, colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    col_chart, _ = st.columns([1, 1])
    with col_chart:
        st.pyplot(fig, width="stretch")

    st.write("")

    sim, sim_reason = simulate_primary_gap(row, pillar_report, elite_features)
    if sim:
        gap_label, _ = METRIC_INFO.get(sim["gap_metric"], (sim["gap_metric"], ""))
        near = min(sim["simulated_time_conservative"], sim["simulated_time_typical"])
        far = max(sim["simulated_time_conservative"], sim["simulated_time_typical"])
        render_html(f'''
        <div class="opportunity-card">
            <div class="bottom-title">🎯 Your Biggest Opportunity</div>
            <div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">
                Your primary gap is <b>{gap_label}</b>. Holding every other metric at your own current
                performance, closing just this one gap to a realistic point within the optimal range —
                not the very best in the sample, just solidly inside it — projects to:
            </div>
            <div class="opportunity-range">{near:.2f}s – {far:.2f}s</div>
            <div style="font-size:11.5px; color:#64748b;">vs. your actual {sim['actual_time']:.2f}s.
                A physics-based simulation, not a guarantee.</div>
        </div>
        ''')

    if biggest_gap_pillars:
        takeaway = (f"{row['name']}'s performance is most limited by "
                    f"{', '.join(biggest_gap_pillars)}. Focused training here offers the biggest opportunity to drop time.")
    else:
        takeaway = f"{row['name']} is performing within or above the optimal range across every metric measured — a well-rounded, strong performance."

    all_gaps = []
    for pname, entries in pillar_report["pillars"].items():
        for e in entries:
            if e["severity"] in ("Gap", "Critical Gap"):
                all_gaps.append((e["severity"], e["metric"]))
    all_gaps.sort(key=lambda x: x[0] != "Critical Gap")
    actions = [ACTION_TEXT.get(m, "") for _, m in all_gaps[:3]]
    actions_html = "".join(f'<div class="action-item">{i+1}. {a}</div>' for i, a in enumerate(actions)) or \
                   '<div class="action-item">No priority actions — every metric is within or above the optimal range.</div>'

    col_a, col_b = st.columns(2)
    with col_a:
        render_html(f'<div class="bottom-card"><div class="bottom-title">🎯 Key Takeaway</div>'
                    f'<div style="font-size:13px; color:#cbd5e1; line-height:1.55;">{takeaway}</div></div>')
    with col_b:
        render_html(f'<div class="bottom-card"><div class="bottom-title">✅ Priority Actions</div>{actions_html}</div>')

    with st.expander("What do Strength / On-par / Gap / Critical Gap mean?"):
        for tier, explanation in VERDICT_EXPLAIN.items():
            st.markdown(f"{badge(tier)} &nbsp; {explanation}", unsafe_allow_html=True)
        st.caption("The 'optimal range' is drawn from a benchmark sample of 20 male and 20 female "
                   "fast 100m Freestyle performances — a reference range, not a ranking.")

else:
    st.info("Enter a new race or load a saved one above to see your diagnostic report.")