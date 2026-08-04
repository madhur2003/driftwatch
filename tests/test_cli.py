"""Tests for CLI argument parsing and top-level error handling."""

from __future__ import annotations

from driftwatch import db
from driftwatch.cli import build_parser, main


def test_log_level_accepted_before_subcommand():
    args = build_parser().parse_args(["--log-level", "DEBUG", "ingest"])
    assert args.log_level == "DEBUG"


def test_log_level_accepted_after_subcommand():
    # The natural form: `driftwatch ingest --log-level DEBUG`.
    args = build_parser().parse_args(["ingest", "--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


def test_log_level_defaults_to_info():
    args = build_parser().parse_args(["ingest"])
    assert args.log_level == "INFO"


def test_ingest_without_api_key_exits_cleanly_and_logs_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("DRIFTWATCH_DB_PATH", str(db_path))
    monkeypatch.delenv("EIA_API_KEY", raising=False)

    # Expected failure: returns 1 (no traceback escapes main()).
    assert main(["ingest", "--lookback-hours", "1"]) == 1

    with db.get_connection(db_path) as conn:
        run = db.recent_runs(conn, limit=1)[0]
        assert run["status"] == "failed"
        assert run["error"]


def test_status_on_empty_db_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_DB_PATH", str(tmp_path / "cli.db"))
    assert main(["status"]) == 0


def test_bootstrap_seeds_and_trains(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("DRIFTWATCH_MODEL_PATH", str(tmp_path / "m.joblib"))

    assert main(["bootstrap", "--demo", "--days", "20"]) == 0
    with db.get_connection(tmp_path / "d.db") as conn:
        assert db.observation_count(conn) > 0
    assert (tmp_path / "m.joblib").exists()
    assert (tmp_path / "m.reference.json").exists()  # drift reference captured


def test_bootstrap_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_DB_PATH", str(tmp_path / "d.db"))
    monkeypatch.setenv("DRIFTWATCH_MODEL_PATH", str(tmp_path / "m.joblib"))

    assert main(["bootstrap", "--demo", "--days", "20"]) == 0
    mtime = (tmp_path / "m.joblib").stat().st_mtime_ns
    # Second run: data present and model present -> no reseed, no retrain.
    assert main(["bootstrap", "--demo", "--days", "20"]) == 0
    assert (tmp_path / "m.joblib").stat().st_mtime_ns == mtime
