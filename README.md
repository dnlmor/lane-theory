# 🏊 Lane Theory
### Data-Driven Optimization of Swimming Race Strategy

**[Live demo →]** *https://s-lanetheory.streamlit.app*

Lane Theory analyzes elite performances to identify what
technique and pacing patterns are associated with fast swimming, then
gives any individual swimmer a personalized diagnostic — their strengths,
their biggest opportunity, and a physically-grounded simulation of what's
achievable by fixing it. It is a **diagnostic and coaching-support tool**,
not a ranking system: the elite dataset exists to define a benchmark
range, never a leaderboard.

---

## What it does

1. **You enter a race** — lap-by-lap splits, breakout distances, stroke
   counts, reaction time.
2. **The app compares you against an optimal-range benchmark** built from
   20 male and 20 female fast (for now I took the 100m Freestyle performances), across 6
   pillars: Pacing & Split Consistency, Underwater Hydrodynamics, Stroke
   Efficiency, Cadence Dynamics.
3. **Every metric is classified on a 5-tier scale** — Excellent, Good,
   Medium, Bad, Critical — never a single ambiguous number.
4. **You get a plain-language coaching report**: what to maintain, what
   to improve, and a physics-based simulation of your primary time-drop
   opportunity — grounded in real distance/speed/time arithmetic, not a
   black-box prediction.

## Why this exists

Most "swim analytics" projects either rank athletes against each other
(not useful to an individual trying to improve) or treat physical
attributes like height as if they determine technique quality (unfair to
swimmers who don't match a "typical" elite build). Lane Theory is built
entirely around **scale-free, self-referential performance ratios** (a
swimmer's own Lap 2 compared to their own Lap 1) and never ranks — every
output is a comparison against a range.

## Methodology

**Data.** 20 male + 20 female 100m Freestyle performances (LCM), selected
as a benchmark sample of well-executed swims — not a leaderboard.

**Feature engineering.** Raw splits are transformed into scale-free
ratios: `pacing_decay_ratio`, `si_retention` (stroke efficiency
retention), `breakout_decay_ratio`, `surface_speed_decay_ratio`,
`intra_lap_fade_ratio` (both laps), and `relative_stroke_length` (stroke
length normalized by height). Full definitions in
[`docs/metric_glossary.md`](docs/metric_glossary.md).

**Modeling.** One Random Forest Regressor per gender, predicting
`final_time_sec` directly from the engineered features. SHAP identifies
which features matter most for each gender's model — used to build the
elite benchmark model in `notebooks/02_modeling_and_optimization.ipynb`.

**Individual diagnosis (the dashboard).** For any swimmer, a **6-Pillar
report** compares their own ratios against the optimal range (min–max +
mean) for their gender, classifying each metric on a 5-tier scale
(Excellent/Good/Medium/Bad/Critical — Good/Medium split the middle
"within range" band so a metric near the top of the range reads
differently from one near the bottom).

**Simulation, not ML prediction.** Random Forest models cannot
extrapolate beyond the time range they were trained on — they will
always predict a time close to the elite range, even for an individual
who is far slower, because a tree-based model can only average similar
training examples. This was caught directly during development. The fix:
individual time simulations use a **deterministic physics-based
simulator** (`simulate_gap_driven_time` / `simulate_underwater_speed_gap`
in `src/optimization.py`) that reconstructs the affected lap using real
distance ÷ speed = time arithmetic, adjusting only the swimmer's actual
flagged gap and holding everything else at their own real performance.
Output is always a range, never a single point.

## Key design decisions (and why they changed)

This project went through several real architectural pivots — each one
driven by catching a genuine problem during development, not assumed
upfront:

1. **Height-based clustering, tried and dropped.** An early version split
   swimmers into height sub-groups before modeling. This caused a real
   extrapolation problem for anyone outside a cluster's observed height
   range, relied on fragile small samples, and made height an
   architectural boundary rather than a normalizing factor. **Fix:**
   height is now used only inside `relative_stroke_length`'s formula; the
   model architecture is gender-only.
2. **Ranking, tried and dropped.** An early version predicted a
   swimmer's rank within the elite group — not useful to an individual,
   and couldn't support meaningful "what's achievable" simulation.
   **Fix:** the model predicts `final_time_sec` directly.
3. **A few metrics' "correct direction" contradicted basic swim physics
   when read purely from the 20-swimmer correlation** (e.g. underwater
   breakout distance appeared to correlate backwards in one gender's
   sample — a small-sample artifact, not real signal). **Fix:** a
   `KNOWN_DIRECTIONS` override hardcodes the physically correct direction
   for unambiguous metrics, while genuinely context-dependent ones
   (stroke rate, breakout %) stay data-driven.
4. **`intra_lap_fade` was originally a raw speed difference, not a
   ratio** — meaning it partly measured overall speed rather than pure
   pacing quality, and produced misleading verdicts for swimmers far from
   the elite ability level. **Fix:** converted to a self-referential
   ratio (`intra_lap_fade_ratio`), consistent with every other decay
   metric in the project.

## Limitations

- **Elite sample size (20 per gender) is small.** Some raw correlations
  contradicted established swim physics — addressed via `KNOWN_DIRECTIONS`
  for unambiguous cases, but the underlying data is still thin.
- **Random Forest models cannot extrapolate** (see Simulation above) —
  this is why individual time simulations never use the RF directly.
- **Scale-free features can mask overall ability level.** Because nearly
  every feature is a self-referential ratio (by design, to avoid
  height/body-size bias), a swimmer with excellent *technique* ratios but
  a much slower *absolute* speed can look artificially "elite-like" on
  paper. The 6-Pillar report remains valid regardless (it's a pure range
  comparison); absolute time simulations are most reliable for swimmers
  whose overall ability is closer to the elite tier.
- **Association, not causation.** All relationships described are
  correlational, from a benchmark sample — not a controlled experiment.
- **No `target_time` / goal-chasing feature**, by design — how
  aggressively to pursue a specific time is a coaching decision, not an
  automated one.
- **A saved-race data-entry safeguard exists but isn't foolproof.** If
  every Lap 2 metric simultaneously outperforms Lap 1 (a rare pattern),
  the dashboard flags this as a possible Lap 1/Lap 2 data-swap — but
  can't confirm it without the raw footage.

## Repository structure

```
lane-theory/
├── data/
│   ├── elite/{raw,processed}/       # benchmark sample (40 races)
│   └── individuals/{raw,processed}/ # any user's own logged races
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_modeling_and_optimization.ipynb
│   └── 03_individual_diagnosis.ipynb
├── src/
│   ├── feature_engineering.py       # shared, imported by all notebooks
│   └── optimization.py              # pillar report, tiering, physics simulator, dispatcher
├── docs/
│   └── metric_glossary.md
├── dashboard/
│   └── app.py                       # Streamlit app — the live diagnostic tool
├── models/                          # saved gender models + SHAP values
├── .streamlit/
│   └── config.toml                  # dark theme
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn (`RandomForestRegressor`), SHAP,
SciPy, Matplotlib, Streamlit.

## Running it locally

```bash
git clone <your-repo-url>
cd lane-theory
pip install -r requirements.txt

# Regenerate the trained models (only needed once, or after changing features):
jupyter notebook notebooks/02_modeling_and_optimization.ipynb   # run all cells

# Launch the dashboard:
streamlit run dashboard/app.py
```

## Roadmap

- [x] Elite dataset collection (benchmark sample, not a leaderboard)
- [x] Scale-free feature engineering
- [x] Gender-separated benchmark modeling + SHAP
- [x] 6-Pillar individual diagnosis, 5-tier classification
- [x] Physics-based gap simulation (avoiding RF extrapolation)
- [x] Streamlit dashboard, deployed
- [x] Generalized dispatcher covering every metric (currently: breakout
      decay, both underwater speeds)
- [ ] 200m Freestyle extension (v1.1)

## Author

Built by **Daniel Siahaan** — MSc AI for Marketing Strategy
student, former competitive swimmer.
