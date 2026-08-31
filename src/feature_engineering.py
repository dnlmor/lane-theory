"""
Lane Theory — Feature Engineering (v3)
Reusable functions to transform raw lap-by-lap swim data into a scale-free,
performance-focused feature set. Designed to run identically on elite
benchmark data (data/elite/) and individual swimmer data (data/individuals/),
so both populations are directly comparable downstream.

Design principles:
- Height is used ONLY as a normalizing factor inside specific formulas
  (relative_stroke_length) — it is never carried forward as a standalone
  model input or used to group/cluster swimmers.
- Prefer self-referential ratios (a swimmer's own Lap 2 vs. their own
  Lap 1) wherever possible — these are scale-free by construction and
  need no height adjustment at all, since the swimmer's own scale cancels
  out automatically.
- This feature set describes HOW a race was executed (technique, pacing,
  efficiency) — it is never used to rank swimmers against each other.
  The elite dataset these features are computed on is a benchmark sample,
  not a leaderboard.

100m Freestyle schema (2 laps): l1_*, l2_*
"""

import pandas as pd
import numpy as np


LAP_DISTANCE = 50.0
QUARTER_DISTANCE = 25.0


def compute_age_at_race(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dob = pd.to_datetime(df["date_of_birth"])
    race_date = pd.to_datetime(df["race_date"])
    df["age_at_race"] = (race_date - dob).dt.days / 365.25
    return df


def compute_lap_features(df: pd.DataFrame, lap: int) -> pd.DataFrame:
    """Compute underwater speed, surface speed, stroke length, relative
    (height-normalized) stroke length, stroke rate, Stroke Index, breakout
    percentage, and SWOLF for a given lap (1 or 2).

    Relative stroke length exists so stroke length is judged fairly across
    different body sizes: a taller swimmer naturally covers more distance
    per stroke due to limb length/wingspan. Height is used purely as a
    normalizing factor here — never as a grouping key.

    Stroke Index (SI = velocity x stroke_length) is a standard swim-science
    efficiency metric — multiplicatively combining speed and stroke length,
    rather than additively combining differently-scaled units the way SWOLF
    does (time in seconds + a raw stroke count).
    """
    df = df.copy()
    p = f"l{lap}_"

    breakout_dist = df[f"{p}breakout_distance"]
    breakout_time = df[f"{p}breakout_time"]
    stroke_count = df[f"{p}stroke_count"]
    total_time = df[f"{p}total_time"]
    height_m = df["height_cm"] / 100

    surface_distance = LAP_DISTANCE - breakout_dist
    surface_time = total_time - breakout_time
    stroke_length = surface_distance / stroke_count
    surface_speed = surface_distance / surface_time

    df[f"{p}underwater_speed"] = breakout_dist / breakout_time
    df[f"{p}surface_distance"] = surface_distance
    df[f"{p}surface_time"] = surface_time
    df[f"{p}surface_speed"] = surface_speed
    df[f"{p}stroke_length"] = stroke_length
    df[f"{p}relative_stroke_length"] = stroke_length / height_m
    df[f"{p}stroke_rate"] = (stroke_count / surface_time) * 60
    df[f"{p}stroke_index"] = surface_speed * stroke_length
    df[f"{p}breakout_pct"] = breakout_dist / LAP_DISTANCE
    df[f"{p}swolf"] = surface_time + stroke_count

    return df


def compute_quarter_split_features(df: pd.DataFrame) -> pd.DataFrame:
    """25m-resolution features within each lap, using l1_split_25m /
    l2_split_25m (lap-relative for l2, per project convention). These are
    self-referential (compare a swimmer's own segments to each other), so
    they're scale-free by construction — no height adjustment needed."""
    df = df.copy()

    # Lap 1: split_25m is cumulative from race start
    l1_first25_time = df["l1_split_25m"]
    l1_second25_time = df["l1_total_time"] - df["l1_split_25m"]
    df["l1_first25_speed"] = QUARTER_DISTANCE / l1_first25_time
    df["l1_second25_speed"] = QUARTER_DISTANCE / l1_second25_time
    df["intra_lap1_fade"] = df["l1_first25_speed"] - df["l1_second25_speed"]

    # Lap 2: split_25m is lap-relative (time from L2 wall push-off to 75m mark)
    l2_first25_time = df["l2_split_25m"]
    l2_second25_time = df["l2_total_time"] - df["l2_split_25m"]
    df["l2_first25_speed"] = QUARTER_DISTANCE / l2_first25_time
    df["l2_second25_speed"] = QUARTER_DISTANCE / l2_second25_time
    df["intra_lap2_fade"] = df["l2_first25_speed"] - df["l2_second25_speed"]

    # Freshest segment (L1 first 25) vs most fatigued segment (L2 last 25)
    df["finish_vs_fresh_ratio"] = df["l2_second25_speed"] / df["l1_first25_speed"]

    return df


def compute_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Race-level features. Pacing Decay, SI Retention, Breakout Decay,
    and Surface Speed Decay are all self-referential ratios (a swimmer's
    own Lap 2 divided by their own Lap 1), so they are scale-free by
    construction and require no height adjustment.
    """
    df = df.copy()

    # Self-referential ratios — scale-free without needing height
    df["pacing_decay_ratio"] = df["l2_total_time"] / df["l1_total_time"]
    df["si_retention"] = df["l2_stroke_index"] / df["l1_stroke_index"]
    df["breakout_decay_ratio"] = df["l2_breakout_distance"] / df["l1_breakout_distance"]
    df["surface_speed_decay_ratio"] = df["l2_surface_speed"] / df["l1_surface_speed"]

    df["front_end_pct"] = df["l1_total_time"] / df["pb_50m_seconds"]

    # Cross-swimmer-relevant deltas — use the height-normalized version
    df["relative_stroke_length_drop"] = (
        df["l1_relative_stroke_length"] - df["l2_relative_stroke_length"]
    )
    df["stroke_rate_change"] = df["l2_stroke_rate"] - df["l1_stroke_rate"]
    df["swolf_change"] = df["l2_swolf"] - df["l1_swolf"]

    # Raw breakout_drop kept for reference/interpretability, but
    # breakout_decay_ratio (above) is the preferred scale-free version
    df["breakout_drop"] = df["l1_breakout_distance"] - df["l2_breakout_distance"]

    return df


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw lap data -> engineered feature set.
    Works identically on elite and individual data.

    Note: height_cm is preserved in the output (carried over from raw
    input) since it's still useful for context/display, but it is NOT
    intended to be used as a standalone model input or grouping key —
    only relative_stroke_length (which is derived FROM height) should
    inform cross-swimmer comparisons.
    """
    df = raw_df.copy()

    df = compute_age_at_race(df)
    df = compute_lap_features(df, lap=1)
    df = compute_lap_features(df, lap=2)
    df = compute_quarter_split_features(df)
    df = compute_race_level_features(df)

    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/elite/raw/100m_freestyle_raw.csv")
    features = engineer_features(raw)
    features.to_csv("data/elite/processed/100m_freestyle_features.csv", index=False)
    print(f"Saved {len(features)} rows, {len(features.columns)} columns")