# Lane Theory
### Data-Driven Optimization of Swimming Race Strategy

Lane Theory analyzes elite 100m Freestyle performances to identify what
technique and pacing patterns are associated with fast swimming, then
gives any individual swimmer a personalized diagnostic — their strengths,
their biggest opportunity, and a physically-grounded simulation of what's
achievable by fixing it. It is a **diagnostic and coaching-support tool**,
not a ranking system: the elite dataset exists to define a benchmark
range, never a leaderboard.

---

## Problem

Swimmers and coaches often know a race "fell apart" somewhere but lack a
systematic way to isolate *where* and *by how much* — and generic
performance-analytics projects tend to either (a) rank athletes against
each other, which isn't useful to an individual trying to improve, or (b)
treat physical attributes like height as if they determine technique
quality, which unfairly penalizes swimmers who don't match a "typical"
elite body type.

Lane Theory addresses both: it is built entirely around **scale-free,
self-referential performance ratios** (a swimmer's own Lap 2 compared to
their own Lap 1) rather than raw numbers or physical comparisons, and it
never ranks — every output is a comparison against an elite **range**.

## Methodology

**Data.** 20 male + 20 female 100m Freestyle performances (LCM), selected
as a benchmark sample of well-executed swims — not a leaderboard. Raw
lap-by-lap splits, breakout distances/times, stroke counts, and reaction
times were collected per swimmer.

**Feature engineering.** Raw data is transformed into scale-free ratios:
`pacing_decay_ratio`, `si_retention`, `breakout_decay_ratio`,
`surface_speed_decay_ratio`, `intra_lap_fade_ratio` (both laps), and
`relative_stroke_length` (stroke length normalized by the swimmer's own
height). Full definitions and swim-coaching interpretation for every
metric are in [`docs/metric_glossary.md`](docs/metric_glossary.md).

**Modeling.** One Random Forest Regressor per gender, predicting
`final_time_sec` directly from the engineered features. SHAP identifies
which features matter most for each gender's model.

**Diagnosis.** For any individual swimmer, a **6-Pillar report** compares
their own ratios against the elite range (min–max + mean) for their
gender, classifying each metric as Strength / On-par / Gap. The pillar
with the most Gaps is surfaced as the primary strategy focus.

**Simulation.** Rather than asking the Random Forest to predict an
individual's time directly (see *Limitations* below for why this fails),
a deterministic physics-based simulator reconstructs the affected lap
using real distance/speed/time arithmetic, adjusting **only the
swimmer's actual flagged gap metric** and holding everything else at
their own real performance. Output is always a conservative/typical/
optimistic range, never a single number.

## Key design decisions (and why they changed)

This project went through two significant architectural pivots, both
driven by catching real problems during development rather than
assuming the first design was right:

1. **Height-based clustering, tried and dropped.** An early version split
   swimmers into height sub-groups per gender before modeling, on the
   theory that body size fundamentally changes technique. This caused a
   real extrapolation problem for any swimmer outside a cluster's
   observed height range, relied on fragile 9–14 row samples, and — more
   importantly — contradicted the project's actual goal, since it made
   height an architectural boundary rather than a normalizing factor.
   **Fix:** height is now used only inside `relative_stroke_length`'s
   formula; the model architecture is gender-only.

2. **Ranking, tried and dropped.** An early version trained the benchmark
   model to predict a swimmer's rank within the elite group. This doesn't
   answer the question an individual swimmer actually has, and doesn't
   support meaningful "what's achievable" simulation. **Fix:** the model
   now predicts `final_time_sec` directly, and every output — pillar
   verdicts and simulations alike — is framed as a range comparison
   against the elite benchmark, never a position.

## Key findings

- Male and female elites show meaningfully different top predictive
  features (SHAP): stroke-length retention dominates for male swimmers,
  late-race fade (`intra_lap2_fade`) dominates for female swimmers —
  supporting the decision to model each gender separately.
- Once normalized by height, male and female elite stroke length looks
  far more similar than raw stroke length suggests — both groups execute
  structurally similar, well-optimized strokes scaled to their own frame.
- A pooled (non-gender-separated) correlation between pacing/efficiency
  ratios and final time showed a spurious relationship, driven entirely
  by the fact that male swimmers are both faster overall and show
  different average ratios — a Simpson's-paradox-style confound that
  further validated the gender-separated modeling choice.

## Limitations

- **Elite sample size (20 per gender) is small.** Some correlations
  computed directly from this data contradicted well-established swim
  physics (e.g. `breakout_decay_ratio`'s raw correlation sign in the male
  sample). Rather than trust noisy correlation blindly, metrics with an
  unambiguous physically-correct direction are hardcoded
  (`KNOWN_DIRECTIONS` in `src/optimization.py`); only genuinely
  trade-off-driven metrics (stroke rate, breakout %) remain data-driven.
- **Random Forest models cannot extrapolate.** A tree-based model's
  prediction is bounded by the range of target values it was trained on
  — it is structurally incapable of predicting a time outside the elite
  group's observed range (46.4–47.3s for male), no matter how different
  an individual's actual inputs are. This was caught directly: an
  out-of-range test swimmer (real time 57.53s) was predicted at ~47s by
  the RF model. **The fix:** individual time predictions/simulations
  never use the RF directly — they use either the swimmer's own real
  recorded time, or the deterministic physics-based simulator described
  above.
- **Scale-free features can mask overall ability level.** Because nearly
  every feature is a self-referential ratio (by design, to avoid
  height/body-size bias), a swimmer with excellent *technique* ratios but
  a much slower *absolute* speed can look artificially "elite-like" on
  paper. The pillar report remains valid regardless (it's a pure range
  comparison), but this is why absolute time predictions are only
  reliable for swimmers whose overall ability is already close to the
  elite tier.
- **Association, not causation.** All relationships described (e.g.
  "longer relative stroke length is associated with faster times") are
  correlational, drawn from a benchmark sample — not a controlled
  experiment. Diagnosis output should be read as "associated with," never
  as a guarantee.
- No `target_time` / goal-chasing feature by design — deciding how
  aggressively to pursue a specific time is left to the swimmer and coach,
  not automated.

## Repository structure

```
lane-theory/
├── data/
│   ├── elite/{raw,processed}/       # benchmark sample
│   └── individuals/{raw,processed}/ # any user's own logged races
├── notebooks/
│   ├── 01_data_and_features.ipynb
│   ├── 02_modeling_and_optimization.ipynb
│   └── 03_individual_diagnosis.ipynb
├── src/
│   ├── feature_engineering.py       # shared, imported by all notebooks
│   └── optimization.py              # shared, imported by notebooks 02 & 03
├── docs/
│   └── metric_glossary.md
├── dashboard/app.py                 # Streamlit (in progress)
├── models/                          # saved gender models + SHAP values
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn (RandomForestRegressor), SHAP,
SciPy (`differential_evolution`), Matplotlib/Seaborn, Streamlit
(dashboard).

## Roadmap

- [x] Elite dataset collection (top-20-per-gender benchmark sample)
- [x] Scale-free feature engineering
- [x] Gender-separated benchmark modeling + SHAP
- [x] 6-Pillar individual diagnosis
- [x] Physics-based gap simulation (avoiding RF extrapolation)
- [ ] Generalized dispatcher (auto-select simulator based on flagged gap)
- [ ] Streamlit dashboard
- [ ] 200m Freestyle extension (v1.1)

## Author

Built by Daniel Siahaan (Haaniel) — MSc AI for Marketing Strategy student,
former competitive swimmer, as a portfolio project.