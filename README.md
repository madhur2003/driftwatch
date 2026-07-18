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

This is being built in weekly increments.

- [x] **Week 1 — Ingestion.** Pull the live EIA feed on a schedule and store it
      cleanly and idempotently. ← *you are here*
- [ ] Week 2 — Train a demand forecaster and serve it behind FastAPI; containerize.
- [ ] Week 3 — Drift layer: PSI + KS tests on incoming features, plus error monitoring.
- [ ] Week 4 — Monitoring dashboard: live predictions, recent error, drift status.
- [ ] Week 5 — Deploy, then deliberately feed it shifted data to prove the flag fires.

## What Week 1 delivers

```
src/driftwatch/
  config.py       env-driven settings (API key, DB path, timeouts)
  eia_client.py   typed EIA v2 client with pagination + retry/backoff
  db.py           SQLite store; idempotent upsert + ingestion run log
  ingest.py       orchestration: resolve window -> fetch -> upsert -> log run
  cli.py          `driftwatch ingest | status | init-db`
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

## Scheduled ingestion

[`.github/workflows/ingest.yml`](.github/workflows/ingest.yml) runs ingestion on
a cron (every 6 hours) using an `EIA_API_KEY` repository secret, and uploads the
refreshed SQLite file as a build artifact. For real cross-run persistence, point
`DRIFTWATCH_DB_PATH` at a hosted Postgres and drop the artifact step.

## Development

```bash
make test    # pytest — client pagination/retry, idempotent upsert, run logging
make lint    # ruff
make fmt     # ruff format
```

The test suite runs fully offline: the EIA client is exercised through a mocked
HTTP transport, so no API key or network is needed to validate the plumbing.

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
