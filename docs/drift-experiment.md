# Drift experiment — proving the flag fires

The point of Driftwatch is not the forecast; it is that the service notices when
its inputs have moved out from under it and says so. This is that test: feed the
running system a deliberate distribution shift and confirm the flag flips from
`OK` to `ALERT`.

Everything below is reproducible in about a minute:

```bash
scripts/drift_demo.sh        # requires `pip install .`
```

## Setup

- **Data:** 45 days of hourly demand for `PJM`. For a no-secrets run this is the
  synthetic generator (`driftwatch synth`), which reproduces the structure real
  demand has — a daily double peak, lower weekends, a slow seasonal swing, and
  noise. The same experiment runs against real EIA data when a key is configured.
- **Model:** `HistGradientBoostingRegressor` on calendar + lag features. On this
  run: **MAPE 2.01%**, **60.7% better than a seasonal-naive baseline** (validation
  MAE 1,348 MWh). Training also captures a *drift reference* — the demand
  distribution the model learned on.

## The experiment

1. Train on the stable 45 days (captures the reference).
2. Drift check on the most recent 7 days → should be `OK`.
3. Inject a **+35% demand level shift** into the most recent 7 days
   (`driftwatch synth --days 7 --shift 0.35`) — a heat-wave-scale jump to a level
   the model never trained on.
4. Drift check again → should be `ALERT`.

## Results

| Signal | Stable window | After +35% shift | Threshold |
| --- | --- | --- | --- |
| **PSI** (input distribution) | 0.037 | **9.07** | warn ≥ 0.10, alert ≥ 0.25 |
| **KS** statistic (p-value) | 0.075 (p = 0.40) | **0.808 (p ≈ 0)** | — |
| **Recent MAE** vs baseline | 1,357 (1.01×) | **21,390 (15.9×)** | warn ≥ 1.5×, alert ≥ 2.0× |
| **Flag** | `OK` | **`ALERT`** | — |

Both independent detectors fired:

- **Input drift.** PSI jumped from 0.037 to 9.07 (two orders of magnitude past
  the alert line) and the KS test rejected "same distribution" at p ≈ 0 — the
  incoming demand simply does not look like the training distribution anymore.
- **Performance decay.** Recent prediction error rose to ~16× the model's
  training-time error. The gradient-boosted trees cannot extrapolate above the
  demand levels they were trained on, so a 35% shift is not something the model
  quietly absorbs — and the monitor sees it.

`driftwatch drift --fail-on-alert` exits non-zero on `ALERT`, so a scheduled job
or CI step pages instead of the model failing silently in production.

## Why this matters

Most "ML app" projects stop at serving a prediction. The interesting failure mode
in production is not a crash — it is a model that keeps returning confident,
wrong answers after the world changes. Driftwatch turns that silent failure into
a visible, thresholded signal: the dashboard banner turns red, the
`drift_reports` table records it, and the exit code trips an alert.

## Limitations (honest notes)

- A pure **level shift** is the easiest kind of drift to catch. Subtler changes
  (a shape change at constant mean, or slow seasonal creep) move PSI/KS far less;
  the error monitor is the better guard there, and both are deliberately combined.
- PSI is computed on demand level only. Calendar features are excluded on purpose
  — their distribution is fixed by the window, not by drift, so monitoring them
  produces false alarms rather than signal.
- Thresholds (PSI 0.10 / 0.25, error ratio 1.5 / 2.0) are the standard rules of
  thumb. In a real deployment they would be tuned against the observed
  false-alarm rate over a few weeks of live data.
