"""
Lane Theory — Optimization & Diagnosis (v3)
Reusable functions for gender-level benchmarking and constrained
optimization, shared across notebooks 02 and 03.

Design principles carried from feature_engineering.py:
- No height-based grouping. Gender is the only population split.
- The elite dataset is a benchmark sample, not a leaderboard — nothing
  here ranks swimmers against each other. Models predict final_time_sec
  directly, and outputs are reported as RANGES (min, max, mean), never a
  single "optimal" number.
- Optimization levers are restricted to primitive (directly measured or
  simple per-lap) features. Derived/ratio features (pacing_decay_ratio,
  si_retention, breakout_decay_ratio, etc.) are valid model inputs but
  are NEVER optimized directly, since they're computed from the
  primitives and moving them independently can produce physically
  inconsistent combinations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


primitive_cols = [
    "l1_reaction_time",
    "l1_underwater_speed", "l2_underwater_speed",
    "l1_relative_stroke_length", "l2_relative_stroke_length",
    "l1_stroke_rate", "l2_stroke_rate",
]

derived_cols = [
    "pacing_decay_ratio", "si_retention",
    "breakout_decay_ratio", "surface_speed_decay_ratio",
    "front_end_pct",
    "relative_stroke_length_drop", "stroke_rate_change",
    "swolf_change", "breakout_drop",
    "intra_lap1_fade", "intra_lap2_fade", "finish_vs_fresh_ratio",
]

feature_cols = primitive_cols + derived_cols


def get_gender_bounds(df, gender, cols):
    """Bounds (min, max) drawn from the full elite group of one gender —
    no height sub-grouping. Returns a dict of {col: (min, max)}."""
    sub = df.loc[df["gender"] == gender, cols]
    return {c: (sub[c].min(), sub[c].max()) for c in cols}


def get_gender_range_summary(df, gender, cols):
    """Full range summary (min, max, mean) for reporting to the user —
    always show mean alongside the range, never present a single point
    as 'the' target."""
    sub = df.loc[df["gender"] == gender, cols]
    return {c: (sub[c].min(), sub[c].max(), sub[c].mean()) for c in cols}

def get_directional_bounds(df, gender, cols, current_row, target_col="final_time_sec"):
    """Restrict each lever's optimization bound to only the direction
    that's actually associated with a FASTER time within this gender's
    elite data — determined via simple correlation sign. This prevents
    the optimizer from recommending a change that contradicts basic swim
    logic (e.g., decreasing underwater speed) due to noise in a small
    sample. The lever can move from the swimmer's current value toward
    the elite bound on the beneficial side only — never past their
    current value in the unhelpful direction."""
    sub = df[df["gender"] == gender]
    bounds = {}
    for c in cols:
        corr = sub[c].corr(sub[target_col])
        elite_min, elite_max = sub[c].min(), sub[c].max()
        current_val = current_row[c]

        if corr <= 0:
            # higher value associated with faster (lower) time -> allow increase only
            lo = current_val
            hi = max(elite_max, current_val)
        else:
            # higher value associated with slower (higher) time -> allow decrease only
            lo = min(elite_min, current_val)
            hi = current_val

        # Guard against degenerate zero-width bounds (current already at the edge)
        if lo == hi:
            lo, hi = min(lo, elite_min), max(hi, elite_max)

        bounds[c] = (lo, hi)
    return bounds

def get_top_levers(gender, shap_values_dict, primitive_cols, feature_cols, threshold=0.3):
    """Primitive features whose mean |SHAP value| is at least `threshold`
    x the top primitive feature's value, for this gender's model."""
    shap_vals = shap_values_dict[gender]
    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_cols)
    primitive_shap = mean_abs_shap[primitive_cols].sort_values(ascending=False)
    cutoff = primitive_shap.iloc[0] * threshold
    return primitive_shap[primitive_shap >= cutoff].index.tolist()


def optimize_swimmer(current_row, model, all_feature_cols, adjustable_cols, bounds_dict):
    """Gradient-free constrained optimization (differential_evolution) —
    required because tree-based models produce piecewise-constant
    predictions, which cause gradient-based optimizers (e.g. SLSQP) to
    report zero gradient and stop immediately without exploring."""
    bounds = [bounds_dict[c] for c in adjustable_cols]
    base_features = current_row[all_feature_cols].copy()

    def objective(x):
        feat = base_features.copy()
        feat[adjustable_cols] = x
        X = pd.DataFrame([feat[all_feature_cols].values], columns=all_feature_cols)
        return model.predict(X)[0]

    result = differential_evolution(objective, bounds, seed=42, maxiter=200, tol=1e-6)
    return result


def diagnose_swimmer(row, features_df, models, shap_values_dict, primitive_cols, feature_cols, threshold=0.3):
    """Run the full diagnosis for one swimmer: predicted current time,
    optimizer-simulated time using elite-associated primitive values, and
    a lever-by-lever breakdown with elite range context. No ranking
    anywhere in the output."""
    gender = row["gender"]

    top_levers = get_top_levers(gender, shap_values_dict, primitive_cols, feature_cols, threshold)
    bounds_dict = get_directional_bounds(features_df, gender, top_levers, row)
    range_summary = get_gender_range_summary(features_df, gender, top_levers)
    model = models[gender]

    result = optimize_swimmer(row, model, feature_cols, top_levers, bounds_dict)

    current_X = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    current_pred = model.predict(current_X)[0]

    report = {
        "name": row["name"],
        "gender": gender,
        "current_predicted_time": current_pred,
        "simulated_predicted_time": result.fun,
        "levers": []
    }
    for lever, current, optimized in zip(top_levers, row[top_levers].values, result.x):
        lo, hi, mean = range_summary[lever]
        report["levers"].append({
            "feature": lever,
            "current": current,
            "simulated": optimized,
            "elite_min": lo,
            "elite_max": hi,
            "elite_mean": mean,
        })
    return report


def print_report(report):
    print(f"{report['name']} ({report['gender']})")
    print(f"Predicted time: {report['current_predicted_time']:.2f}s -> "
          f"simulated: {report['simulated_predicted_time']:.2f}s")
    print()
    for lv in report["levers"]:
        print(f"  {lv['feature']}: {lv['current']:.3f} -> {lv['simulated']:.3f}  "
              f"(elite range: {lv['elite_min']:.3f}-{lv['elite_max']:.3f}, mean {lv['elite_mean']:.3f})")