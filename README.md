# Driftwatch

**A self-monitoring prediction service that flags its own decay.**

Most "ML app" projects stop at serving predictions. Driftwatch forecasts
regional electricity demand *and* watches its own inputs and errors, so it can
say when it is becoming untrustworthy instead of quietly rotting. The point
isn't the model — it's operating a system that knows when it's drifting.

The data feed is the [EIA open API](https://www.eia.gov/opendata/): hourly
electricity demand by balancing authority. Demand has strong daily and seasonal
structure plus weather-driven shifts, so the drift layer has something real to
catch.

---

## Status

Built in weekly increments — all five complete.

- [x] **Week 1 — Ingestion.** Pull the live EIA feed on a schedule and store it
      cleanly and idempotently.
- [x] **Week 2 — Forecast & serve.** Train a demand forecaster and serve it
      behind FastAPI; containerize.
- [x] **Week 3 — Drift layer.** PSI + KS on the demand distribution and
      prediction-error monitoring, with the threshold that raises the flag.
- [x] **Week 4 — Dashboard.** A live monitoring page — the flag, model quality,
      the forecast, and the drift-history timeline — served by the API itself.
- [x] **Week 5 — Deploy & stress.** A self-bootstrapping container ([Docker →
      any free tier](docs/DEPLOY.md)) and a reproducible experiment that feeds
      the running service shifted data to [prove the flag fires](docs/drift-experiment.md).

## What's inside

```
src/driftwatch/
  config.py       env-driven settings (API key, DB + model paths, timeouts)
  eia_client.py   typed EIA v2 client with pagination + retry/backoff
  db.py           SQLite store; idempotent upsert + ingestion run log
  ingest.py       orchestration: resolve window -> fetch -> upsert -> log run
  features.py     calendar + lag features; one path for train and inference
  synthetic.py    realistic offline demand generator (demos + tests)
  model.py        train / evaluate / persist / predict (gradient boosting)
  drift.py        PSI + KS distribution drift, prediction-error decay, flag
  dashboard.py    self-contained monitoring page + its aggregate data endpoint
  api.py          FastAPI service: dashboard, /health, /model, /predict, /drift
  cli.py          ingest | status | init-db | synth | train | predict | serve | drift
```

Design choices worth noting:

- **Idempotent storage.** Observations are keyed on
  `(respondent, data_type, period_utc)`, so re-running ingestion over an
  overlapping window refreshes values in place instead of duplicating rows —
  scheduled jobs can safely overlap.
- **Robust fetching.** Transient failures (network errors, 5xx, HTTP 429) are
  retried with exponential backoff; permanent errors (bad key, bad request)
  fail fast. Large windows are paged transparently.
- **An operational trail.** Every run is recorded in `ingestion_runs` (window,
  rows written, status, error) — the raw material for the monitoring view later.
- **No train/serve skew.** Training and inference build features through the
  *same* `features.py` code path, and every feature reads only history older
  than the target hour — so forecasts up to 24h ahead are leakage-free and the
  API can build features for future hours from stored observations alone.
- **Honest evaluation.** The model is scored on a time-ordered hold-out (never
  a shuffled split) against a seasonal-naive "same hour yesterday" baseline, so
  its reported skill is real.
- **Self-monitoring.** Training snapshots a drift *reference* (the demand
  distribution it learned on). Each drift check scores a recent window against
  it — PSI + KS for input shift, recent error vs. the training baseline for
  decay — and raises `ok` / `warn` / `alert`. That flag is the whole point: the
  system says when it is becoming untrustworthy instead of failing silently.

## Quickstart

```bash
# 1. Get a free EIA API key: https://www.eia.gov/opendata/register.php
cp .env.example .env
# then edit .env and set EIA_API_KEY

# 2. Install so the `driftwatch` command is on your PATH
pip install .

# 3. Pull the last 72h of PJM demand, then see what landed
driftwatch ingest --lookback-hours 72
driftwatch status
```

Prefer not to install? Run it straight from source (this is also the dev loop):

```bash
make ingest            # or: PYTHONPATH=src python -m driftwatch ingest --lookback-hours 72
make status            # or: PYTHONPATH=src python -m driftwatch status
```

Backfill a specific window (UTC):

```bash
driftwatch ingest --respondent PJM --start 2024-06-01 --end 2024-06-07
```

Track more balancing authorities (repeat the flag):

```bash
driftwatch ingest --respondent PJM --respondent NYIS --lookback-hours 48
```

## Forecast & serve

The forecaster is a gradient-boosted regressor
(scikit-learn `HistGradientBoostingRegressor`) over calendar features (cyclical
hour / day-of-week / day-of-year, weekend flag) and demand lags (24h, 48h, 168h)
plus the prior day's mean. The model is deliberately not the hard part — the
point is a trained artifact served behind an API that reports its own quality.

**Try the whole loop offline — no API key needed** (synthetic data stands in for
a live feed):

```bash
driftwatch synth --days 60        # seed a realistic synthetic demand series
driftwatch train                  # -> val MAE / RMSE / MAPE + skill vs baseline
driftwatch predict --horizon 24   # next-24h forecast from the CLI
driftwatch serve                  # FastAPI on http://127.0.0.1:8000  (docs at /docs)
```

Then hit the service:

```bash
curl localhost:8000/health
curl localhost:8000/model                     # metrics + training window + features
curl -X POST localhost:8000/predict \
     -H 'content-type: application/json' \
     -d '{"respondent":"PJM","horizon_hours":6}'
```

| Endpoint             | Description                                              |
| -------------------- | ------------------------------------------------------- |
| `GET /`              | The monitoring dashboard (HTML)                          |
| `GET /dashboard/data`| Aggregate JSON powering the dashboard                    |
| `GET /health`        | Liveness + whether a model is loaded and when it trained |
| `GET /model`         | Training metadata: metrics, baseline, features, window   |
| `POST /predict`      | Forecast by `horizon_hours` (≤24) or explicit `periods`  |
| `POST /drift`        | Score a recent window for drift; records the result      |
| `GET /drift/history` | Recent drift reports (the monitoring timeline)           |

Training persists three files next to each other under `models/` by default
(`DRIFTWATCH_MODEL_PATH`): `model.joblib` (the estimator), `model.json` (metrics
+ metadata), and `model.reference.json` (the drift reference).

## Drift detection

The self-monitoring layer scores a recent window against the reference captured
at training time and raises `ok` / `warn` / `alert`:

- **Input drift** — Population Stability Index (PSI) and a Kolmogorov-Smirnov
  two-sample test on the demand distribution. PSI ≥ 0.1 warns, ≥ 0.25 alerts.
  (Calendar features are intentionally not monitored: their distribution is
  fixed by the window, not by real-world drift.)
- **Prediction-error decay** — the model's recent error vs. the error it
  achieved at training time. A ratio ≥ 1.5 warns, ≥ 2.0 alerts.

Prove the flag fires — no API key needed (or run [`scripts/drift_demo.sh`](scripts/drift_demo.sh)):

```bash
driftwatch synth --days 45 && driftwatch train   # reference captured here
driftwatch drift                                  # -> status: OK

driftwatch synth --days 7 --shift 0.35            # inject a +35% level shift
driftwatch drift --fail-on-alert                  # -> status: ALERT, exits 2
```

That +35% shift takes PSI from 0.04 to ~9 and recent error to ~16× the training
baseline — full numbers and interpretation in
[docs/drift-experiment.md](docs/drift-experiment.md).

Every check is written to a `drift_reports` table (and surfaced at
`GET /drift/history`) — the operational trail the dashboard renders.
`--fail-on-alert` makes the command exit non-zero, so a cron job or CI step can
page on it.

## Dashboard

The service serves its own monitoring page at `/` — a single self-contained
HTML page (no external assets) that polls `GET /dashboard/data` and shows:

- the **drift flag** as a prominent `ok` / `warn` / `alert` banner;
- stat tiles: model MAPE, skill vs. baseline, current PSI, recent-error ratio;
- a **forecast chart** — the last 72h of actual demand and the next 24h predicted;
- the **drift-history timeline** — every recorded check, newest first.

```bash
driftwatch synth --days 45 && driftwatch train   # give it data + a model
driftwatch serve                                 # open http://127.0.0.1:8000
```

It auto-refreshes every 30s and is theme-aware (light/dark). Seed a shift
(`driftwatch synth --days 7 --shift 0.35`) and watch the banner turn red.

## Docker & deploy

The image (Python 3.12 slim) serves the API + dashboard with uvicorn and binds
to `$PORT`. With `DRIFTWATCH_BOOTSTRAP_DEMO=1` it self-provisions a synthetic
dataset and trains a model on start, so a fresh container has a live dashboard
with no volume or API key:

```bash
docker build -t driftwatch .
docker run --rm -p 8000:8000 -e DRIFTWATCH_BOOTSTRAP_DEMO=1 driftwatch
# open http://localhost:8000
```

For real data, mount a populated store instead of bootstrapping:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" driftwatch
```

Free-tier deploy (Render blueprint, Fly.io, Hugging Face Spaces) is documented in
[docs/DEPLOY.md](docs/DEPLOY.md).

## Scheduled ingestion

[`.github/workflows/ingest.yml`](.github/workflows/ingest.yml) runs ingestion on
a cron (every 6 hours) using an `EIA_API_KEY` repository secret, and uploads the
refreshed SQLite file as a build artifact. For real cross-run persistence, point
`DRIFTWATCH_DB_PATH` at a hosted Postgres and drop the artifact step.

## Development

```bash
make test    # pytest — ingestion, features/leakage, model skill, drift, API
make lint    # ruff
make fmt     # ruff format
```

The test suite runs fully offline: the EIA client is exercised through a mocked
HTTP transport, the model trains on a synthetic series, and the API is driven via
FastAPI's `TestClient` — so no API key or network is needed to validate anything.

### Environment note: editable installs on Python 3.14 + macOS

`pip install -e .` (editable/development install) can silently produce an
**un-importable** package on Python 3.14 under macOS. Cause: pip marks its
`__editable__*.pth` marker files with the macOS `UF_HIDDEN` flag, and Python
3.14's `site` module now skips any `.pth` file carrying that flag — so the path
to `src/` never lands on `sys.path`. This affects *any* editable install on such
a machine, not just this project.

Driftwatch sidesteps it entirely: a plain `pip install .` copies the package in
(no `.pth`, no problem), and the dev/test loop runs from source via
`PYTHONPATH=src` (`make ingest`, `make test`), which needs no install at all.

## Data source

EIA API v2, `electricity/rto/region-data` — hourly `value` for demand (`D`) and
related types (`DF`, `NG`, `TI`) per balancing authority, reported in UTC.

## License

MIT
