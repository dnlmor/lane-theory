# Lane Theory — Metric Glossary

Every metric below is compared against an **elite range** (min–max + mean)
drawn from a benchmark sample of 20 male and 20 female fast 100m
Freestyle performances — never against a ranking or leaderboard position.
Where a metric has a well-established "correct" direction from swim
science, that direction is hardcoded (see `KNOWN_DIRECTIONS` in
`src/optimization.py`); where a metric involves a genuine trade-off, the
direction is left data-driven.

---

## Pillar 1 — Pacing Decay

**`pacing_decay_ratio`** = Lap 2 total time ÷ Lap 1 total time

How much the whole second half slows down relative to the first. A value
near the elite mean (~1.09 male, ~1.07 female) reflects the standard,
expected fade even among elites — some slowdown is normal and not a
weakness. Much higher means over-aggressive early pacing or a real
fatigue problem; much lower (close to 1.0, even pacing) is good, but
worth checking it isn't a sign of going out *too* conservatively and
leaving time on the table.

*Direction: data-driven within elite sample (context-dependent).*

---

## Pillar 2 — Segmental Split Consistency

**`intra_lap1_fade_ratio`, `intra_lap2_fade_ratio`** = second-25m speed ÷
first-25m speed, within a single lap

How well a swimmer holds speed *within* one 50m, isolated from the wall
turn. Closer to 1.0 means minimal internal slowdown; lower means real
fade even before reaching the wall. Splitting this by lap distinguishes
"fading gradually across the whole race" from "specifically collapsing in
the final 25m."

*Direction: higher is better (hardcoded — less within-lap fade is
essentially always beneficial).*

**`finish_vs_fresh_ratio`** = final-25m speed ÷ first-25m speed
(race-wide)

Compares the most fatigued moment of the race to the freshest. The
single cleanest number for "how much do you have left at the very end."

*Direction: higher is better (hardcoded).*

---

## Pillar 3 — Underwater Hydrodynamics

**`breakout_decay_ratio`** = Lap 2 breakout distance ÷ Lap 1 breakout
distance

How much underwater distance survives into the second wall push-off.
High means strong underwater endurance/technique holding up under
fatigue; low is the classic "gasping for the surface early" pattern
after a turn.

*Direction: higher is better (hardcoded — more underwater distance is
essentially always beneficial, since underwater phases have less drag
than surface swimming; a data-driven check on this project's own 20-male
sample actually found the opposite correlation, which was a small-sample
artifact, not real signal — see README limitations).*

**`l{n}_breakout_pct`** = breakout distance ÷ 50m

How much of each lap is spent underwater. More is generally good up to
the legal limit (15m), but this genuinely trades off against surface
technique quality — pushing too far underwater without strong kick
technique isn't automatically faster.

*Direction: data-driven (genuine trade-off, no universal answer).*

**`l{n}_underwater_speed`** = breakout distance ÷ breakout time

Raw push-off/kick speed underwater. Higher reflects a more powerful,
better-executed underwater phase (dolphin kick strength and technique).

*Direction: higher is better (hardcoded).*

---

## Pillar 4 — Stroke Mechanics & Water Grip

**`si_retention`** = Lap 2 Stroke Index ÷ Lap 1 Stroke Index (Stroke
Index = surface speed × stroke length)

The single best "does technique hold up under fatigue" number. Near or
above 1.0 is exceptional — technique doesn't degrade under load; well
below means the stroke is falling apart late in the race (arms spinning
without grip on the water).

*Direction: higher is better (hardcoded).*

**`l{n}_relative_stroke_length`** = (surface distance ÷ stroke count) ÷
swimmer's height in meters

How much "reach and grip" per pull, independent of body size — this is
what makes stroke length comparable across a 168cm swimmer and a 203cm
swimmer fairly. Higher means a longer, more efficient pull; low can mean
a short, choppy stroke.

*Direction: higher is better (hardcoded).*

---

## Pillar 5 — Cadence

**`l{n}_stroke_rate`** = (stroke count ÷ surface time) × 60, in
cycles/minute

Arm turnover speed. This is the one metric with no universal "higher is
better" — it only means something paired with stroke length. High rate +
short stroke length is the "spinning your wheels" panic pattern; high
rate + maintained stroke length is a legitimate elite sprint strategy.

*Direction: data-driven (genuine trade-off — must be read alongside
stroke length, not in isolation).*

---

## Pillar 6 — Surface Pacing Engine

**`surface_speed_decay_ratio`** = Lap 2 surface speed ÷ Lap 1 surface
speed (excludes underwater phases on both laps)

Isolates whether the "clean swimming engine" itself is fading, separate
from turn/underwater issues. High means surface-swimming fitness holds
up; a swimmer who fades a lot overall (high `pacing_decay_ratio`) but
keeps this high is being told the real problem lives elsewhere (turns,
underwater), not raw swimming fitness.

*Direction: higher is better (hardcoded).*

---

## Verdict logic

Every metric gets classified **Strength / On-par / Gap** by comparing the
swimmer's value against the elite range for their gender:

- **Strength**: at or beyond the elite mean, in the beneficial direction
- **On-par**: within the elite min–max range, but not yet at the mean
- **Gap**: outside the elite range entirely, in the unfavorable direction

The pillar with the most Gap verdicts is surfaced as the primary strategy
focus — never a ranking, always a plain-language "where's the biggest
opportunity" summary.