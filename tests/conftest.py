"""Shared fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from driftwatch.eia_client import DemandRecord


@pytest.fixture
def sample_records() -> list[DemandRecord]:
    base = datetime(2024, 6, 1, 0, tzinfo=UTC)
    return [
        DemandRecord(
            respondent="PJM",
            respondent_name="PJM Interconnection, LLC",
            period_utc=base.replace(hour=h),
            data_type="D",
            value=85000.0 - h * 500,
            value_units="megawatthours",
        )
        for h in range(3)
    ]
