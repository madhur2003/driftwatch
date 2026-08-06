# Deploying Driftwatch

Driftwatch ships as a single Docker image that serves the API and the dashboard.
With `DRIFTWATCH_BOOTSTRAP_DEMO=1` the container **self-provisions** a synthetic
dataset and trains a model on cold start, so a fresh deploy has a live dashboard
immediately — no API key, no mounted volume, no manual steps. That makes it a
good fit for free tiers with an ephemeral filesystem.

The container binds to `$PORT` (default 8000) and exposes `GET /health` for
health checks.

## Render (recommended)

One click (reads [`render.yaml`](../render.yaml); you'll sign in and approve):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/madhur2003/driftwatch)

Or manually — [`render.yaml`](../render.yaml) is a ready blueprint.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. It reads `render.yaml`,
   builds the Dockerfile, and sets `DRIFTWATCH_BOOTSTRAP_DEMO=1`.
3. Open the service URL — the dashboard is at `/`.

Render injects `$PORT` and polls `/health`. The free plan spins down when idle
and cold-starts on the next request (the bootstrap re-runs, which is fine).

## Fly.io

Docker-native and supports a volume, so the SQLite store can persist across
restarts (skip the volume for a pure demo).

```bash
fly launch --no-deploy          # generates fly.toml from the Dockerfile
fly secrets set DRIFTWATCH_BOOTSTRAP_DEMO=1
# optional persistence:
fly volumes create driftwatch_data --size 1
#   then mount it at /app in fly.toml and point DB/model paths inside it
fly deploy
```

## Hugging Face Spaces (Docker)

Create a **Docker** Space and push this repo. Spaces serve on `$PORT` (7860) —
the entrypoint already respects `$PORT`. Set `DRIFTWATCH_BOOTSTRAP_DEMO=1` in the
Space **Settings → Variables**.

## Local Docker

```bash
docker build -t driftwatch .
docker run --rm -p 8000:8000 -e DRIFTWATCH_BOOTSTRAP_DEMO=1 driftwatch
# open http://localhost:8000
```

To serve real data instead of the demo, mount a populated store and omit the
bootstrap flag:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" driftwatch
```

## Real data instead of the demo

Leave `DRIFTWATCH_BOOTSTRAP_DEMO` unset and feed the service real EIA data:

- set `EIA_API_KEY` (a free key from <https://www.eia.gov/opendata/register.php>),
- run ingestion on a schedule (see [`.github/workflows/ingest.yml`](../.github/workflows/ingest.yml)),
- and `driftwatch train` once enough history has accumulated.

## Stress-testing the deployed service

To reproduce the drift experiment against a running deployment, open a shell in
the container (`render shell`, `fly ssh console`, or the Spaces terminal) and run:

```bash
driftwatch drift                          # baseline: OK
driftwatch synth --days 7 --shift 0.35    # inject a +35% level shift
driftwatch drift                          # now: ALERT
```

Refresh the dashboard and the banner turns red. See
[drift-experiment.md](drift-experiment.md) for the full write-up and numbers.
