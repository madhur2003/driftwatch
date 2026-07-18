.PHONY: help install dev test lint fmt ingest status clean
PY ?= python3

# Run the CLI straight from source: no install step, always reflects your edits,
# and immune to the Python 3.14 + pip editable-install quirk (see README).
RUN := PYTHONPATH=src $(PY) -m driftwatch

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install as a real package so the `driftwatch` command is on PATH
	$(PY) -m pip install .

dev:  ## Install dev + test tooling (run the CLI with `make ingest`, tests with `make test`)
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff
	$(PY) -m ruff check .

fmt:  ## Auto-format with ruff
	$(PY) -m ruff format .

ingest:  ## Pull the last 72h of demand (needs EIA_API_KEY)
	$(RUN) ingest --lookback-hours 72

status:  ## Show what has been ingested
	$(RUN) status

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
