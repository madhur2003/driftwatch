"""Tests for the SQLite store: schema, idempotent upsert, run logs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from driftwatch import db


def test_upsert_inserts_records(tmp_path, sample_records):
    with db.get_connection(tmp_path / "d.db") as conn:
        written = db.upsert_records(conn, sample_records)
        assert written == len(sample_records)
        assert db.observation_count(conn) == len(sample_records)


def test_upsert_is_idempotent(tmp_path, sample_records):
    with db.get_connection(tmp_path / "d.db") as conn:
        db.upsert_records(conn, sample_records)
        db.upsert_records(conn, sample_records)  # re-run identical window
        # No duplicate rows — the primary key collapses them.
        assert db.observation_count(conn) == len(sample_records)


def test_upsert_refreshes_value_on_conflict(tmp_path, sample_records):
    with db.get_connection(tmp_path / "d.db") as conn:
        db.upsert_records(conn, sample_records)
        revised = [replace(r, value=(r.value or 0) + 1) for r in sample_records]
        db.upsert_records(conn, revised)

        assert db.observation_count(conn) == len(sample_records)
        row = conn.execute(
            "SELECT value FROM demand_observations WHERE respondent='PJM' AND period_utc=?",
            (sample_records[0].period_utc.isoformat(),),
        ).fetchone()
        assert row["value"] == (sample_records[0].value or 0) + 1


def test_run_log_lifecycle(tmp_path):
    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = datetime(2024, 6, 1, 3, tzinfo=UTC)
    with db.get_connection(tmp_path / "d.db") as conn:
        run_id = db.start_run(conn, ["PJM"], ["D"], start, end)
        db.finish_run(conn, run_id, rows_upserted=42, status="success")

        run = db.recent_runs(conn, limit=1)[0]
        assert run["status"] == "success"
        assert run["rows_upserted"] == 42
        assert run["finished_at_utc"] is not None


def test_latest_period(tmp_path, sample_records):
    with db.get_connection(tmp_path / "d.db") as conn:
        db.upsert_records(conn, sample_records)
        latest = db.latest_period(conn, "PJM", "D")
        assert latest == sample_records[-1].period_utc.isoformat()
