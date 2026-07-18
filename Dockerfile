# Serve the Driftwatch forecasting API.
# Uses Python 3.12 (stable, wide wheel coverage) regardless of the host's version.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DRIFTWATCH_DB_PATH=/app/data/driftwatch.db \
    DRIFTWATCH_MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The database and model artifact are provided at run time via mounts, e.g.:
#   docker run -p 8000:8000 \
#     -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" driftwatch
EXPOSE 8000

CMD ["uvicorn", "driftwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
