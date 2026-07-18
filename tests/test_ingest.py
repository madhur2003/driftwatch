"""Tests for ingestion orchestration and window resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from driftwatch import db, ingest
from driftwatch.config import Settings
from driftwatch.ingest import resolve_window, run_ingestion


def _settings(tmp_path) -> Settings:
    return Settings(
        eia_api_key="test-key",
        db_path=tmp_path / "d.db",
        request_timeout=5.0,
        max_retries=2,
    )


def test_resolve_window_explicit_range():
    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = datetime(2024, 6, 2, tzinfo=UTC)
    assert resolve_window(start, end, None) == (start, end)


def test_resolve_window_default_lookback():
    start, end = resolve_window(None, None, None)
    assert (end - start) == timedelta(hours=ingest.DEFAULT_LOOKBACK_HOURS)
    assert start.tzinfo == UTC


def test_resolve_window_rejects_inverted_range():
    start = datetime(2024, 6, 2, tzinfo=UTC)
    end = datetime(2024, 6, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        resolve_window(start, end, None)


class _FakeClient:
    """Stands in for EIAClient; returns canned records, no network."""

    def __init__(self, records, *args, **kwargs):
        self._records = records

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def fetch_demand(self, *args, **kwargs):
        return self._records


def test_run_ingestion_stores_and_logs_success(tmp_path, monkeypatch, sample_records):
    monkeypatch.setattr(ingest, "EIAClient", lambda *a, **k: _FakeClient(sample_records))
    result = run_ingestion(
        _settings(tmp_path),
        respondents=["PJM"],
        data_types=["D"],
        start=datetime(2024, 6, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, 2, tzinfo=UTC),
    )

    assert result.records_fetched == len(sample_records)
    assert result.rows_upserted == len(sample_records)

    with db.get_connection(tmp_path / "d.db") as conn:
        assert db.observation_count(conn) == len(sample_records)
        run = db.recent_runs(conn, limit=1)[0]
        assert run["status"] == "success"


def test_run_ingestion_marks_run_failed_and_reraises(tmp_path, monkeypatch):
    class _BoomClient(_FakeClient):
        def fetch_demand(self, *args, **kwargs):
            raise RuntimeError("EIA is down")

    monkeypatch.setattr(ingest, "EIAClient", lambda *a, **k: _BoomClient([]))

    with pytest.raises(RuntimeError, match="EIA is down"):
        run_ingestion(
            _settings(tmp_path),
            respondents=["PJM"],
            data_types=["D"],
            lookback_hours=24,
        )

    with db.get_connection(tmp_path / "d.db") as conn:
        run = db.recent_runs(conn, limit=1)[0]
        assert run["status"] == "failed"
        assert "EIA is down" in run["error"]
