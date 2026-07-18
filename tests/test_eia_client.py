"""Tests for the EIA client: parsing, pagination, and retry behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from driftwatch.eia_client import (
    EIAClient,
    EIAError,
    parse_eia_period,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eia_region_data.json"


def _client(handler, **kwargs) -> EIAClient:
    return EIAClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        backoff_min=0.0,
        backoff_max=0.0,
        **kwargs,
    )


def test_parse_eia_period():
    assert parse_eia_period("2024-06-01T05") == datetime(2024, 6, 1, 5, tzinfo=UTC)


def test_missing_api_key_raises():
    with pytest.raises(EIAError):
        EIAClient("")


def test_fetch_parses_and_coerces_null_value():
    payload = json.loads(FIXTURE.read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _client(handler) as client:
        records = client.fetch_demand(
            ["PJM"], datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, 2, tzinfo=UTC)
        )

    assert len(records) == 3
    assert records[0].value == 85000.0
    assert records[0].respondent_name == "PJM Interconnection, LLC"
    # The null value in the fixture must coerce to None, not crash.
    assert records[2].value is None


def test_pagination_follows_offset():
    # total = 3, page_length = 2 -> two pages (2 rows, then 1 row).
    rows = [
        {"period": f"2024-06-01T{h:02d}", "respondent": "PJM", "type": "D", "value": str(1000 + h)}
        for h in range(3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        length = int(request.url.params.get("length", "5000"))
        page = rows[offset : offset + length]
        return httpx.Response(200, json={"response": {"total": "3", "data": page}})

    with _client(handler, page_length=2) as client:
        records = client.fetch_demand(
            ["PJM"], datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, 2, tzinfo=UTC)
        )

    assert [r.value for r in records] == [1000.0, 1001.0, 1002.0]


def test_retries_on_server_error_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json={"response": {"total": "0", "data": []}})

    with _client(handler, max_retries=3) as client:
        records = client.fetch_demand(
            ["PJM"], datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, 1, tzinfo=UTC)
        )

    assert records == []
    assert calls["n"] == 2  # one failure, one success


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="invalid api key")

    with _client(handler, max_retries=4) as client:
        with pytest.raises(EIAError):
            client.fetch_demand(
                ["PJM"], datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, 1, tzinfo=UTC)
            )

    assert calls["n"] == 1  # 4xx fails fast, no retries
