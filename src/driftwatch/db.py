"""SQLite storage for demand observations and ingestion run logs.

The upsert is idempotent on (respondent, data_type, period_utc), so re-running
ingestion over an overlapping window never creates duplicates — it refreshes
values in place. The ingestion_runs table records every run for the operational
view built in later weeks.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .eia_client import DemandRecord

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS demand_observations (
    respondent       TEXT NOT NULL,
    period_utc       TEXT NOT NULL,
    data_type        TEXT NOT NULL,
    value            REAL,
    value_units      TEXT,
    respondent_name  TEXT,
    ingested_at_utc  TEXT NOT NULL,
    PRIMARY KEY (respondent, data_type, period_utc)
);

CREATE INDEX IF NOT EXISTS idx_demand_period
    ON demand_observations (period_utc);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc    TEXT NOT NULL,
    finished_at_utc   TEXT,
    respondents       TEXT,
    data_types        TEXT,
    window_start_utc  TEXT,
    window_end_utc    TEXT,
    rows_upserted     INTEGER DEFAULT 0,
    status            TEXT NOT NULL,
    error             TEXT
);
"""

_UPSERT_SQL = """
INSERT INTO demand_observations
    (respondent, period_utc, data_type, value, value_units, respondent_name, ingested_at_utc)
VALUES
    (:respondent, :period_utc, :data_type, :value, :value_units, :respondent_name, :ingested_at_utc)
ON CONFLICT(respondent, data_type, period_utc) DO UPDATE SET
    value           = excluded.value,
    value_units     = excluded.value_units,
    respondent_name = excluded.respondent_name,
    ingested_at_utc = excluded.ingested_at_utc;
"""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection, creating the parent directory if needed."""
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield an initialized connection, closing it on exit."""
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def upsert_records(conn: sqlite3.Connection, records: Sequence[DemandRecord]) -> int:
    """Insert or refresh observations. Returns the number of rows written."""
    if not records:
        return 0
    ingested_at = _utcnow_iso()
    rows = [
        {
            "respondent": r.respondent,
            "period_utc": r.period_utc.astimezone(UTC).isoformat(),
            "data_type": r.data_type,
            "value": r.value,
            "value_units": r.value_units,
            "respondent_name": r.respondent_name,
            "ingested_at_utc": ingested_at,
        }
        for r in records
    ]
    before = conn.total_changes
    conn.executemany(_UPSERT_SQL, rows)
    conn.commit()
    return conn.total_changes - before


def start_run(
    conn: sqlite3.Connection,
    respondents: Sequence[str],
    data_types: Sequence[str],
    window_start: datetime,
    window_end: datetime,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingestion_runs
            (started_at_utc, respondents, data_types, window_start_utc, window_end_utc, status)
        VALUES (?, ?, ?, ?, ?, 'running')
        """,
        (
            _utcnow_iso(),
            ",".join(respondents),
            ",".join(data_types),
            window_start.isoformat(),
            window_end.isoformat(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    rows_upserted: int,
    status: str = "success",
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingestion_runs
        SET finished_at_utc = ?, rows_upserted = ?, status = ?, error = ?
        WHERE id = ?
        """,
        (_utcnow_iso(), rows_upserted, status, error, run_id),
    )
    conn.commit()


def observation_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM demand_observations").fetchone()[0])


def select_observations(
    conn: sqlite3.Connection,
    respondent: str,
    data_type: str = "D",
    start: str | None = None,
    end: str | None = None,
) -> list[tuple[str, float | None]]:
    """Return (period_utc, value) pairs ordered by time. ISO period strings sort chronologically."""
    query = (
        "SELECT period_utc, value FROM demand_observations WHERE respondent = ? AND data_type = ?"
    )
    params: list[object] = [respondent, data_type]
    if start is not None:
        query += " AND period_utc >= ?"
        params.append(start)
    if end is not None:
        query += " AND period_utc <= ?"
        params.append(end)
    query += " ORDER BY period_utc"
    return [(row[0], row[1]) for row in conn.execute(query, params).fetchall()]


def latest_period(conn: sqlite3.Connection, respondent: str, data_type: str = "D") -> str | None:
    row = conn.execute(
        "SELECT MAX(period_utc) FROM demand_observations WHERE respondent = ? AND data_type = ?",
        (respondent, data_type),
    ).fetchone()
    return row[0] if row else None


def recent_runs(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
