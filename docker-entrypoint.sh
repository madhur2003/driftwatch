#!/usr/bin/env sh
# Container entrypoint: optionally self-provision a demo, then serve the API.
#
# On free tiers with an ephemeral filesystem this runs on every cold start, so
# the demo bootstrap is idempotent — it only seeds/trains when the store is
# empty. For real data, leave DRIFTWATCH_BOOTSTRAP_DEMO unset and mount a
# populated data/ + models/ (or ingest from EIA on a schedule).
set -eu

if [ "${DRIFTWATCH_BOOTSTRAP_DEMO:-0}" = "1" ]; then
  echo "[entrypoint] bootstrapping demo data + model (DRIFTWATCH_BOOTSTRAP_DEMO=1)…"
  driftwatch bootstrap --demo --days "${DRIFTWATCH_DEMO_DAYS:-45}" || \
    echo "[entrypoint] bootstrap failed; serving with whatever is present"
fi

echo "[entrypoint] starting uvicorn on port ${PORT:-8000}"
exec uvicorn driftwatch.api:app --host 0.0.0.0 --port "${PORT:-8000}"
