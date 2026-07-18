"""Ingestion orchestration: resolve a time window, fetch, store, log the run."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from . import db
from .config import Settings
from .eia_client import EIAClient

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_HOURS = 72


@dataclass(frozen=True)
class IngestResult:
    respondents: tuple[str, ...]
    data_types: tuple[str, ...]
    window_start: datetime
    window_end: datetime
    records_fetched: int
    rows_upserted: int


def _floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def resolve_window(
    start: datetime | None,
    end: datetime | None,
    lookback_hours: int | None,
) -> tuple[datetime, datetime]:
    """Resolve an explicit range, or a rolling lookback window ending now (UTC)."""
    end_dt = _floor_hour(end) if end else _floor_hour(datetime.now(UTC))
    if start is not None:
        start_dt = _floor_hour(start)
    else:
        hours = lookback_hours if lookback_hours is not None else DEFAULT_LOOKBACK_HOURS
        start_dt = end_dt - timedelta(hours=hours)
    if start_dt > end_dt:
        raise ValueError(f"start ({start_dt.isoformat()}) is after end ({end_dt.isoformat()})")
    return start_dt, end_dt


def run_ingestion(
    settings: Settings,
    respondents: Sequence[str],
    data_types: Sequence[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    lookback_hours: int | None = None,
) -> IngestResult:
    """Fetch demand for the resolved window and upsert it, logging the run."""
    window_start, window_end = resolve_window(start, end, lookback_hours)
    respondents = tuple(respondents)
    data_types = tuple(data_types)

    with db.get_connection(settings.db_path) as conn:
        run_id = db.start_run(conn, respondents, data_types, window_start, window_end)
        try:
            with EIAClient(
                settings.eia_api_key,
                timeout=settings.request_timeout,
                max_retries=settings.max_retries,
            ) as client:
                records = client.fetch_demand(respondents, window_start, window_end, data_types)
            rows = db.upsert_records(conn, records)
            db.finish_run(conn, run_id, rows_upserted=rows, status="success")
            logger.info(
                "ingest ok: fetched %d records, wrote %d rows for %s [%s .. %s]",
                len(records),
                rows,
                ",".join(respondents),
                window_start.isoformat(),
                window_end.isoformat(),
            )
            return IngestResult(
                respondents=respondents,
                data_types=data_types,
                window_start=window_start,
                window_end=window_end,
                records_fetched=len(records),
                rows_upserted=rows,
            )
        except Exception as exc:
            db.finish_run(conn, run_id, rows_upserted=0, status="failed", error=str(exc))
            logger.error("ingest failed: %s", exc)
            raise
