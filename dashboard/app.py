"""
Lane Theory — Dashboard
Streamlit app: input a race, get a 6-Pillar diagnostic report plus a
physics-based simulation of the swimmer's biggest opportunity. No
ranking anywhere — every comparison is against the elite range for the
swimmer's own gender.

Run with: streamlit run dashboard/app.py
"""

import sys
import os
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
    primitive_cols, feature_cols,
    compute_pillar_report, find_biggest_gap_pillar,
    simulate_primary_gap, print_gap_simulation,
    PILLARS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

INDIVIDUALS_RAW_PATH = os.path.join(DATA_DIR, "individuals", "raw", "individuals_raw.csv")
ELITE_FEATURES_PATH = os.path.join(DATA_DIR, "elite", "processed", "100m_freestyle_features.csv")

VERDICT_COLOR = {"Strength": "#2E7D32", "On-par": "#F9A825", "Gap": "#C62828"}
VERDICT_SCORE = {"Strength": 1.0, "On-par": 0.5, "Gap": 0.0}

RAW_COLUMNS = [
    "athlete_id", "race_id", "name", "gender", "height_cm", "date_of_birth",
    "race_date", "final_time_sec", "pb_50m_seconds", "l1_reaction_time",
    "l1_breakout_distance", "l1_breakout_time", "l1_stroke_count", "l1_total_time",
    "l1_split_25m", "l2_breakout_distance", "l2_breakout_time", "l2_stroke_count",
    "l2_total_time", "l2_split_25m",
]


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


# ---------- UI ----------

st.set_page_config(page_title="Lane Theory", layout="wide")
st.title("🏊 Lane Theory")
st.caption("Data-Driven Optimization of Swimming Race Strategy — a personal diagnostic, "
           "not a leaderboard. Every comparison is against the elite benchmark range for your gender.")

elite_features = load_elite_features()
models = load_models()
shap_values_dict = load_shap_values()
individuals_raw = load_individuals_raw()

tab_new, tab_existing = st.tabs(["New race entry", "View a saved race"])

# ---------- Tab 1: new race entry ----------
with tab_new:
    st.subheader("Enter a race")

    with st.form("new_race_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name")
            gender = st.selectbox("Gender", ["male", "female"])
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0)
            dob = st.date_input("Date of birth", value=date(2005, 1, 1))
        with col2:
            race_date_input = st.date_input("Race date", value=date.today())
            final_time_sec = st.number_input("Final time (seconds)", min_value=40.0, max_value=180.0, value=57.53)
            pb_50m_seconds = st.number_input("50m PB (seconds)", min_value=18.0, max_value=60.0, value=26.5)
            l1_reaction_time = st.number_input("L1 reaction time (s)", min_value=0.3, max_value=1.5, value=0.67)
        with col3:
            l1_split_25m = st.number_input("L1 split @ 25m (cumulative, s)", min_value=8.0, max_value=40.0, value=12.3)
            l2_split_25m = st.number_input("L2 split @ 75m (lap-relative, s)", min_value=8.0, max_value=40.0, value=14.2)

        st.markdown("**Lap 1 (0–50m)**")
        c1, c2, c3, c4 = st.columns(4)
        l1_breakout_distance = c1.number_input("Breakout distance (m)", min_value=0.0, max_value=15.0, value=12.0, key="l1bd")
        l1_breakout_time = c2.number_input("Breakout time (s)", min_value=0.5, max_value=10.0, value=4.6, key="l1bt")
        l1_stroke_count = c3.number_input("Stroke count", min_value=1, max_value=60, value=37, key="l1sc")
        l1_total_time = c4.number_input("Lap total time (s)", min_value=10.0, max_value=60.0, value=27.81, key="l1tt")

        st.markdown("**Lap 2 (50–100m)**")
        c5, c6, c7, c8 = st.columns(4)
        l2_breakout_distance = c5.number_input("Breakout distance (m)", min_value=0.0, max_value=15.0, value=6.5, key="l2bd")
        l2_breakout_time = c6.number_input("Breakout time (s)", min_value=0.5, max_value=10.0, value=2.7, key="l2bt")
        l2_stroke_count = c7.number_input("Stroke count", min_value=1, max_value=60, value=45, key="l2sc")
        l2_total_time = c8.number_input("Lap total time (s)", min_value=10.0, max_value=60.0, value=29.72, key="l2tt")

        submitted = st.form_submit_button("Run diagnosis")

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

        st.success(f"Saved as {race_id}. Showing diagnosis below.")

        row_df = engineer_features(pd.DataFrame([new_row]))
        st.session_state["diagnosis_row"] = row_df.iloc[0]

# ---------- Tab 2: view a saved race ----------
with tab_existing:
    st.subheader("Select a saved race")
    if individuals_raw.empty:
        st.info("No races saved yet — add one in the 'New race entry' tab.")
    else:
        race_id_choice = st.selectbox("Race", individuals_raw["race_id"].tolist())
        if st.button("Load this race"):
            selected_raw = individuals_raw[individuals_raw["race_id"] == race_id_choice]
            row_df = engineer_features(selected_raw)
            st.session_state["diagnosis_row"] = row_df.iloc[0]


# ---------- Diagnosis display ----------

def render_pillar_score_radar(pillar_report):
    labels = list(pillar_report["pillars"].keys())
    scores = []
    for pillar, entries in pillar_report["pillars"].items():
        avg_score = np.mean([VERDICT_SCORE[e["verdict"]] for e in entries])
        scores.append(avg_score)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, scores_plot, color="#1565C0", linewidth=2)
    ax.fill(angles_plot, scores_plot, color="#1565C0", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["Gap", "On-par", "Strength"], fontsize=8)
    ax.set_title("Pillar Overview (avg. verdict per pillar)", pad=20)
    return fig


if "diagnosis_row" in st.session_state:
    row = st.session_state["diagnosis_row"]

    st.divider()
    st.header(f"{row['name']} — Diagnostic Report")
    st.caption(f"{row['gender'].capitalize()} · Actual time: {row['final_time_sec']:.2f}s "
               f"· Compared against the {row['gender']} elite benchmark range")

    pillar_report = compute_pillar_report(row, elite_features)
    biggest_gap = find_biggest_gap_pillar(pillar_report)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.pyplot(render_pillar_score_radar(pillar_report))

    with col_right:
        if biggest_gap:
            st.warning(f"**Primary strategy focus:** {', '.join(biggest_gap)}")
        else:
            st.success("No Gap metrics found — at or above the elite benchmark on every measured dimension.")

        sim, reason = simulate_primary_gap(row, pillar_report, elite_features)
        if sim:
            st.markdown("### Simulated opportunity")
            st.metric("Actual time", f"{sim['actual_time']:.2f}s")
            fastest = min(sim["simulated_time_conservative"], sim["simulated_time_optimistic"])
            slowest = max(sim["simulated_time_conservative"], sim["simulated_time_optimistic"])
            st.metric("Simulated range", f"{fastest:.2f}s – {slowest:.2f}s",
                       delta=f"{sim['actual_time'] - sim['simulated_time_typical']:.2f}s (typical estimate)")
            st.caption(f"Fixing **{sim['gap_metric']}** only — every other metric held at your own "
                       f"actual, current performance. Not a guarantee — a physics-based simulation "
                       f"of what's achievable if this specific gap is closed.")
        else:
            st.info(reason)

    st.markdown("### Full 6-Pillar Breakdown")
    for pillar_name, entries in pillar_report["pillars"].items():
        with st.expander(pillar_name, expanded=bool(biggest_gap and pillar_name in biggest_gap)):
            for e in entries:
                color = VERDICT_COLOR[e["verdict"]]
                st.markdown(
                    f"**{e['metric']}**: {e['value']:.3f} "
                    f"&nbsp;<span style='color:{color}; font-weight:bold'>[{e['verdict']}]</span>&nbsp; "
                    f"(elite range: {e['elite_min']:.3f}–{e['elite_max']:.3f}, mean {e['elite_mean']:.3f})",
                    unsafe_allow_html=True,
                )
else:
    st.info("Enter a new race or load a saved one to see the diagnosis.")