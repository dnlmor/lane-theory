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
    "intra_lap1_fade_ratio", "intra_lap2_fade_ratio", "finish_vs_fresh_ratio",
]

feature_cols = primitive_cols + derived_cols


# For most metrics, "which direction is beneficial" is determined from
# the elite data's correlation with final_time_sec — but for a handful of
# metrics, the correct direction is well-established swim science that a
# noisy 20-row correlation should NOT be allowed to override (we caught a
# real case of this: breakout_decay_ratio's correlation sign in this
# specific 20-swimmer male sample contradicted basic physics — more
# underwater distance is essentially always beneficial, since underwater
# phases have less drag than surface swimming). These are hardcoded;
# everything else falls back to the data-driven correlation sign.
KNOWN_DIRECTIONS = {
    "breakout_decay_ratio": "higher",
    "l1_underwater_speed": "higher",
    "l2_underwater_speed": "higher",
    "si_retention": "higher",
    "pacing_decay_ratio": "lower",
    "surface_speed_decay_ratio": "higher",
    "intra_lap1_fade_ratio": "higher",
    "intra_lap2_fade_ratio": "higher",
    "finish_vs_fresh_ratio": "higher",
    "l1_relative_stroke_length": "higher",
    "l2_relative_stroke_length": "higher",
}


def get_direction(metric, sub, target_col="final_time_sec"):
    """Return 'higher' or 'lower' — which direction of this metric is
    associated with a FASTER time. Checks KNOWN_DIRECTIONS first; falls
    back to the data-driven correlation sign for anything not in that
    list (metrics with genuine trade-offs, e.g. stroke rate, breakout_pct,
    where there's no single obviously-correct direction)."""
    if metric in KNOWN_DIRECTIONS:
        return KNOWN_DIRECTIONS[metric]
    corr = sub[metric].corr(sub[target_col])
    return "higher" if corr <= 0 else "lower"


PILLARS = {
    "Pacing Decay": ["pacing_decay_ratio"],
    "Segmental Split Consistency": ["intra_lap1_fade_ratio", "intra_lap2_fade_ratio", "finish_vs_fresh_ratio"],
    "Underwater Hydrodynamics": ["breakout_decay_ratio", "l1_breakout_pct", "l2_breakout_pct",
                                  "l1_underwater_speed", "l2_underwater_speed"],
    "Stroke Mechanics & Water Grip": ["si_retention", "l1_relative_stroke_length", "l2_relative_stroke_length"],
    "Cadence": ["l1_stroke_rate", "l2_stroke_rate"],
    "Surface Pacing Engine": ["surface_speed_decay_ratio"],
}


def classify_metric(value, elite_min, elite_max, elite_mean, direction):
    """Classify a swimmer's value on one metric as Strength / On-par / Gap,
    relative to the elite range. `direction` is 'higher' or 'lower' —
    which direction is beneficial, from get_direction()."""
    if direction == "higher":
        if value >= elite_mean:
            return "Strength"
        elif value >= elite_min:
            return "On-par"
        else:
            return "Gap"
    else:
        if value <= elite_mean:
            return "Strength"
        elif value <= elite_max:
            return "On-par"
        else:
            return "Gap"


def compute_pillar_report(row, elite_df, target_col="final_time_sec"):
    """Build the 6-Pillar diagnostic report for one swimmer: for every
    metric in every pillar, compare the swimmer's own value against the
    elite range (min, max, mean) for their gender, and classify it as a
    Strength, On-par, or Gap. No ranking anywhere — every comparison is
    against a range, not a position."""
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]

    report = {"name": row["name"], "gender": gender, "pillars": {}}

    for pillar_name, metrics in PILLARS.items():
        pillar_entries = []
        for m in metrics:
            direction = get_direction(m, sub, target_col)
            elite_min, elite_max, elite_mean = sub[m].min(), sub[m].max(), sub[m].mean()
            value = row[m]
            verdict = classify_metric(value, elite_min, elite_max, elite_mean, direction)
            pillar_entries.append({
                "metric": m,
                "value": value,
                "elite_min": elite_min,
                "elite_max": elite_max,
                "elite_mean": elite_mean,
                "verdict": verdict,
            })
        report["pillars"][pillar_name] = pillar_entries

    return report


def print_pillar_report(report):
    print(f"{report['name']} ({report['gender']}) — 6-Pillar Diagnostic Report")
    print("=" * 70)
    for pillar_name, entries in report["pillars"].items():
        print(f"\n{pillar_name}")
        for e in entries:
            print(f"  {e['metric']}: {e['value']:.3f}  [{e['verdict']}]  "
                  f"(elite range: {e['elite_min']:.3f}-{e['elite_max']:.3f}, mean {e['elite_mean']:.3f})")


def find_biggest_gap_pillar(report):
    """Identify the pillar with the most 'Gap' verdicts, as the primary
    strategy focus — a plain-language summary, never a ranking."""
    gap_counts = {
        pillar: sum(1 for e in entries if e["verdict"] == "Gap")
        for pillar, entries in report["pillars"].items()
    }
    max_gaps = max(gap_counts.values())
    if max_gaps == 0:
        return None
    return [p for p, c in gap_counts.items() if c == max_gaps]


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
    current value in the unhelpful direction.

    If a swimmer's current value already sits at or beyond the beneficial
    edge of the elite range, there is genuinely no room to improve that
    lever — it is dropped from the bounds dict entirely (NOT widened back
    to the full range, which would silently undo the directional
    guardrail)."""
    sub = df[df["gender"] == gender]
    bounds = {}
    for c in cols:
        direction = get_direction(c, sub, target_col)
        elite_min, elite_max = sub[c].min(), sub[c].max()
        current_val = current_row[c]

        if direction == "higher":
            lo, hi = current_val, max(elite_max, current_val)
        else:
            lo, hi = min(elite_min, current_val), current_val

        if hi - lo > 1e-9:
            bounds[c] = (lo, hi)
        # else: no room in the beneficial direction — omit this lever

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


def simulate_gap_driven_time(row, elite_df, gap_metric="breakout_decay_ratio"):
    """Deterministic physics-based simulation — NOT a Random Forest
    prediction. This avoids the RF extrapolation trap entirely: instead
    of asking a model trained on 20 elite times (46.40-47.33s for male)
    to predict a time for someone far outside that range — which it
    cannot do, since a Random Forest can only average training examples
    and is structurally bounded by the training target range — this
    reconstructs Lap 2 using real distance/speed/time arithmetic.

    Only the swimmer's actual GAP metric is adjusted (never a metric
    already flagged Strength for them) — everything else uses the
    swimmer's own real, current performance. This avoids double-counting
    improvement across multiple dimensions simultaneously, and keeps the
    simulation grounded in the swimmer's own actual capability.

    Currently supports breakout_decay_ratio as the gap metric, since it
    cleanly maps to a real geometry change (longer underwater -> shorter
    required surface distance) without needing to assume a coupled
    speed/stroke-length target. Returns a LOW/MID/HIGH range (using the
    elite min/mean/max for the gap metric), never a single point.
    """
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]

    actual_l1_total = row["l1_total_time"]
    actual_l1_breakout = row["l1_breakout_distance"]
    actual_l2_underwater_speed = row["l2_underwater_speed"]
    actual_l2_surface_speed = row["l2_surface_speed"]

    elite_min = sub[gap_metric].min()
    elite_mean = sub[gap_metric].mean()
    elite_max = sub[gap_metric].max()

    results = {}
    for label, decay_target in [("conservative", elite_min), ("typical", elite_mean), ("optimistic", elite_max)]:
        sim_breakout = actual_l1_breakout * decay_target
        sim_surface_dist = 50.0 - sim_breakout
        sim_underwater_time = sim_breakout / actual_l2_underwater_speed
        sim_surface_time = sim_surface_dist / actual_l2_surface_speed
        sim_l2_total = sim_underwater_time + sim_surface_time
        results[label] = actual_l1_total + sim_l2_total

    return {
        "name": row["name"],
        "gender": gender,
        "actual_time": row["final_time_sec"],
        "gap_metric": gap_metric,
        "gap_metric_current": row[gap_metric],
        "gap_metric_elite_range": (elite_min, elite_max, elite_mean),
        "simulated_time_conservative": results["conservative"],
        "simulated_time_typical": results["typical"],
        "simulated_time_optimistic": results["optimistic"],
    }


def print_gap_simulation(sim):
    print(f"{sim['name']} ({sim['gender']}) — Gap-Driven Time Simulation")
    print(f"Actual time: {sim['actual_time']:.2f}s")
    print(f"Fixing: {sim['gap_metric']} (current: {sim['gap_metric_current']:.3f}, "
          f"elite range: {sim['gap_metric_elite_range'][0]:.3f}-{sim['gap_metric_elite_range'][1]:.3f}, "
          f"mean {sim['gap_metric_elite_range'][2]:.3f})")
    fastest = min(sim["simulated_time_conservative"], sim["simulated_time_optimistic"])
    slowest = max(sim["simulated_time_conservative"], sim["simulated_time_optimistic"])
    print(f"Simulated time range: {fastest:.2f}s (best case) - {slowest:.2f}s (conservative case), "
          f"typical estimate: {sim['simulated_time_typical']:.2f}s")
    print("Note: this holds every other metric at the swimmer's own actual, "
          "current performance — only the flagged gap metric is adjusted.")


def simulate_underwater_speed_gap(row, elite_df, lap=1):
    """Deterministic physics-based simulation for an underwater-speed gap
    (same non-extrapolating approach as simulate_gap_driven_time). Holds
    breakout distance and surface performance at the swimmer's own actual
    values — only the underwater speed for the given lap is adjusted
    toward the elite range, which changes how long that breakout phase
    takes without touching anything else."""
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]
    metric = f"l{lap}_underwater_speed"

    actual_l1_total = row["l1_total_time"]
    actual_l2_total = row["l2_total_time"]
    actual_breakout_dist = row[f"l{lap}_breakout_distance"]
    actual_breakout_time = row[f"l{lap}_breakout_time"]
    actual_lap_total = row[f"l{lap}_total_time"]
    actual_surface_time = actual_lap_total - actual_breakout_time

    elite_min, elite_mean, elite_max = sub[metric].min(), sub[metric].mean(), sub[metric].max()

    results = {}
    for label, speed_target in [("conservative", elite_min), ("typical", elite_mean), ("optimistic", elite_max)]:
        sim_breakout_time = actual_breakout_dist / speed_target
        sim_lap_total = sim_breakout_time + actual_surface_time  # surface performance held fixed

        if lap == 1:
            results[label] = sim_lap_total + actual_l2_total
        else:
            results[label] = actual_l1_total + sim_lap_total

    return {
        "name": row["name"],
        "gender": gender,
        "actual_time": row["final_time_sec"],
        "gap_metric": metric,
        "gap_metric_current": row[metric],
        "gap_metric_elite_range": (elite_min, elite_max, elite_mean),
        "simulated_time_conservative": results["conservative"],
        "simulated_time_typical": results["typical"],
        "simulated_time_optimistic": results["optimistic"],
    }


SIMULATOR_DISPATCH = {
    "breakout_decay_ratio": lambda row, elite_df: simulate_gap_driven_time(row, elite_df, gap_metric="breakout_decay_ratio"),
    "l1_underwater_speed": lambda row, elite_df: simulate_underwater_speed_gap(row, elite_df, lap=1),
    "l2_underwater_speed": lambda row, elite_df: simulate_underwater_speed_gap(row, elite_df, lap=2),
}


def simulate_primary_gap(row, pillar_report, elite_df):
    """Automatically dispatch to the correct physics-based simulator for
    whichever metric is actually flagged as this swimmer's primary Gap —
    so the dashboard never has to hardcode which simulation to run for a
    given user. Scans the pillar(s) with the most Gap verdicts first, and
    within those, picks the first Gap metric that has a registered
    simulator. If no flagged Gap metric has a simulator built yet, returns
    None with an explanatory reason rather than silently doing nothing or
    guessing."""
    biggest_gap_pillars = find_biggest_gap_pillar(pillar_report)
    if not biggest_gap_pillars:
        return None, "No Gap metrics found — this swimmer is at or above the elite range on every measured dimension."

    candidate_metrics = []
    for pillar in biggest_gap_pillars:
        for entry in pillar_report["pillars"][pillar]:
            if entry["verdict"] == "Gap":
                candidate_metrics.append(entry["metric"])

    for metric in candidate_metrics:
        if metric in SIMULATOR_DISPATCH:
            sim = SIMULATOR_DISPATCH[metric](row, elite_df)
            return sim, None

    return None, (f"Gap(s) found in {candidate_metrics}, but no physics-based simulator "
                   f"is built for these yet — see the 6-Pillar report for the diagnosis, "
                   f"and src/optimization.py's SIMULATOR_DISPATCH to add one.")


def check_extrapolation(row, elite_df, primitive_cols, min_features_outside=1):
    """Random Forest models cannot extrapolate beyond the range of target
    values (final_time_sec) seen during training — every prediction is an
    average of similar training examples, so it is mathematically bounded
    by the training data's time range regardless of how far outside the
    elite technique profile an individual's actual inputs fall.

    A deeper, related issue: this project's feature set is built almost
    entirely from scale-free RATIOS (pacing_decay_ratio, si_retention,
    relative stroke length, etc.) — deliberately, to avoid height/body-size
    bias. A side effect is that these ratios can look elite-like even for
    a swimmer who is fundamentally slower in absolute terms, since nothing
    in the feature set directly encodes raw speed/power. This means the
    predicted/simulated TIME is only reliable for swimmers whose overall
    ability is already close to the elite tier — for anyone meaningfully
    slower overall, the 6-Pillar range comparison remains valid, but the
    absolute time prediction should not be trusted.

    Triggers when at least `min_features_outside` primitive features fall
    outside the elite range for this swimmer's gender.

    Returns (is_extrapolated, fraction_of_levers_outside_range)."""
    gender = row["gender"]
    sub = elite_df[elite_df["gender"] == gender]

    outside_count = 0
    for c in primitive_cols:
        elite_min, elite_max = sub[c].min(), sub[c].max()
        if row[c] < elite_min or row[c] > elite_max:
            outside_count += 1

    fraction_outside = outside_count / len(primitive_cols)
    return outside_count >= min_features_outside, fraction_outside


def diagnose_swimmer(row, features_df, models, shap_values_dict, primitive_cols, feature_cols, threshold=0.3):
    """Run the full diagnosis for one swimmer: predicted current time,
    optimizer-simulated time using elite-associated primitive values, and
    a lever-by-lever breakdown with elite range context. No ranking
    anywhere in the output.

    Levers with no room to improve in the beneficial direction (per
    get_directional_bounds) are excluded from optimization and reported
    separately, rather than silently dropped or forced with fabricated
    bounds."""
    gender = row["gender"]

    is_extrapolated, fraction_outside = check_extrapolation(row, features_df, primitive_cols)

    top_levers = get_top_levers(gender, shap_values_dict, primitive_cols, feature_cols, threshold)
    bounds_dict = get_directional_bounds(features_df, gender, top_levers, row)
    range_summary = get_gender_range_summary(features_df, gender, top_levers)
    model = models[gender]

    optimizable_levers = [lv for lv in top_levers if lv in bounds_dict]
    already_optimal_levers = [lv for lv in top_levers if lv not in bounds_dict]

    current_X = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    current_pred = model.predict(current_X)[0]

    report = {
        "name": row["name"],
        "gender": gender,
        "current_predicted_time": current_pred,
        "is_extrapolated": is_extrapolated,
        "fraction_outside": fraction_outside,
        "levers": [],
        "already_optimal_levers": [],
    }

    if optimizable_levers:
        result = optimize_swimmer(row, model, feature_cols, optimizable_levers, bounds_dict)
        report["simulated_predicted_time"] = result.fun
        for lever, current, optimized in zip(optimizable_levers, row[optimizable_levers].values, result.x):
            lo, hi, mean = range_summary[lever]
            report["levers"].append({
                "feature": lever,
                "current": current,
                "simulated": optimized,
                "elite_min": lo,
                "elite_max": hi,
                "elite_mean": mean,
            })
    else:
        # No lever has room to improve in the beneficial direction
        report["simulated_predicted_time"] = current_pred

    for lever in already_optimal_levers:
        lo, hi, mean = range_summary[lever]
        report["already_optimal_levers"].append({
            "feature": lever,
            "current": row[lever],
            "elite_min": lo,
            "elite_max": hi,
            "elite_mean": mean,
        })

    return report


def print_report(report):
    print(f"{report['name']} ({report['gender']})")

    if report.get("is_extrapolated"):
        print(f"NOTE: {report['fraction_outside']*100:.0f}% of this swimmer's primitive "
              f"features fall outside the elite range. Random Forest models cannot "
              f"extrapolate beyond their training data's time range, and this project's "
              f"features are mostly scale-free ratios that can look elite-like even at a "
              f"much slower absolute speed. The predicted/simulated TIME below should NOT "
              f"be read as a real forecast — rely on the 6-Pillar report (range comparison) "
              f"for a meaningful diagnosis instead.\n")

    print(f"Predicted time: {report['current_predicted_time']:.2f}s -> "
          f"simulated: {report['simulated_predicted_time']:.2f}s")
    print()
    if report["levers"]:
        print("Levers with room to improve:")
        for lv in report["levers"]:
            print(f"  {lv['feature']}: {lv['current']:.3f} -> {lv['simulated']:.3f}  "
                  f"(elite range: {lv['elite_min']:.3f}-{lv['elite_max']:.3f}, mean {lv['elite_mean']:.3f})")
    if report["already_optimal_levers"]:
        print("\nAlready at/beyond the elite-associated edge (no directional room):")
        for lv in report["already_optimal_levers"]:
            print(f"  {lv['feature']}: {lv['current']:.3f}  "
                  f"(elite range: {lv['elite_min']:.3f}-{lv['elite_max']:.3f}, mean {lv['elite_mean']:.3f})")