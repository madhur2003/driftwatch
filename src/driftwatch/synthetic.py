"""Generate a realistic synthetic hourly-demand series for offline demos and tests.

Deterministic given a seed. It reproduces the structure a real demand feed has —
a daily double peak (morning and a larger evening ramp), lower weekends, a slow
seasonal swing, and noise — which is enough for the forecaster to learn and for
later drift experiments (Week 5) to perturb.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from .eia_client import DemandRecord


def _floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def synthetic_records(
    *,
    respondent: str = "PJM",
    hours: int = 24 * 60,
    end: datetime | None = None,
    base: float = 90_000.0,
    noise: float = 0.02,
    seed: int = 0,
) -> list[DemandRecord]:
    """Return ``hours`` of hourly DemandRecords ending at ``end`` (default: now, UTC)."""
    rng = np.random.default_rng(seed)
    end_hour = _floor_hour(end) if end else _floor_hour(datetime.now(UTC))
    periods = [end_hour - timedelta(hours=h) for h in range(hours)][::-1]

    records: list[DemandRecord] = []
    for period in periods:
        h = period.hour
        dow = period.weekday()
        doy = period.timetuple().tm_yday

        morning = np.exp(-0.5 * ((h - 8) / 2.5) ** 2)
        evening = np.exp(-0.5 * ((h - 19) / 2.8) ** 2)
        daily = 0.75 + 0.18 * morning + 0.28 * evening
        weekly = 0.90 if dow >= 5 else 1.0
        seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        value = base * daily * weekly * seasonal * (1.0 + rng.normal(0.0, noise))

        records.append(
            DemandRecord(
                respondent=respondent,
                respondent_name=f"{respondent} (synthetic)",
                period_utc=period,
                data_type="D",
                value=round(float(value), 1),
                value_units="megawatthours",
            )
        )
    return records
