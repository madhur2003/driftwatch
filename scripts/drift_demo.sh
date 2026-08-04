#!/usr/bin/env sh
# Reproducible drift stress test.
#
# A stable window reads OK; a +35% demand level shift then trips the flag to
# ALERT (both the PSI/KS input-drift detector and the prediction-error monitor).
# Uses a throwaway DB/model under $TMPDIR so it never touches your real store.
#
# Requires the `driftwatch` command on PATH (`pip install .`). From a source
# checkout without installing, run instead:
#   PYTHONPATH=src python -m driftwatch <args>   (or use the Makefile targets)
set -eu

WORK="${TMPDIR:-/tmp}/driftwatch-demo"
rm -rf "$WORK"
mkdir -p "$WORK/models"
export DRIFTWATCH_DB_PATH="$WORK/driftwatch.db"
export DRIFTWATCH_MODEL_PATH="$WORK/models/model.joblib"

run() { echo "+ driftwatch $*"; driftwatch "$@"; }

echo "== 1. seed 45 days of stable demand and train (captures the drift reference) =="
run synth --days 45
run train

echo
echo "== 2. drift check on stable data (expect: OK) =="
run drift

echo
echo "== 3. inject a +35% demand level shift into the most recent week =="
run synth --days 7 --shift 0.35

echo
echo "== 4. drift check after the shift (expect: ALERT, exit code 2) =="
if run drift --fail-on-alert; then
  echo "UNEXPECTED: drift did not flag the shift" >&2
  exit 1
else
  echo
  echo "Flag fired as expected (exit code 2). The system caught its own decay."
fi
