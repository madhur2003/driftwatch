# Serve the Driftwatch forecasting API.
# Uses Python 3.12 (stable, wide wheel coverage) regardless of the host's version.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DRIFTWATCH_DB_PATH=/app/data/driftwatch.db \
    DRIFTWATCH_MODEL_PATH=/app/models/model.joblib \
    PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# The entrypoint optionally bootstraps a demo dataset + model (set
# DRIFTWATCH_BOOTSTRAP_DEMO=1), then serves the API on $PORT. For real data,
# mount a populated data/ + models/ (or set EIA_API_KEY and ingest on a schedule).
EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
