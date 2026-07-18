"""A small, typed client over the EIA v2 electricity region-data endpoint.

Handles the two things that actually make ingestion fragile: pagination over
large windows, and transient network / 5xx / rate-limit failures (retried with
exponential backoff). Permanent errors (bad key, malformed request) fail fast.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import EIA_API_BASE

logger = logging.getLogger(__name__)

# EIA hourly periods look like "2024-06-01T05" and are expressed in UTC.
_EIA_PERIOD_FORMAT = "%Y-%m-%dT%H"


class EIAError(RuntimeError):
    """A permanent error from the EIA API (bad request, bad key, bad payload)."""


class EIARetryableError(EIAError):
    """A transient error worth retrying (network failure, 5xx, or rate limit)."""


@dataclass(frozen=True)
class DemandRecord:
    """One observation from the region-data feed."""

    respondent: str
    respondent_name: str | None
    period_utc: datetime
    data_type: str
    value: float | None
    value_units: str | None


def parse_eia_period(period: str) -> datetime:
    """Parse an EIA hourly period string into a timezone-aware UTC datetime."""
    dt = datetime.strptime(period, _EIA_PERIOD_FORMAT)
    return dt.replace(tzinfo=UTC)


def _coerce_value(raw: object) -> float | None:
    """EIA sends values as strings, and occasionally null for missing hours."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class EIAClient:
    """Thin synchronous client over the EIA v2 electricity region-data endpoint."""

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = EIA_API_BASE,
        timeout: float = 30.0,
        max_retries: int = 4,
        page_length: int = 5000,
        backoff_min: float = 1.0,
        backoff_max: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise EIAError(
                "An EIA API key is required. Register for a free key at "
                "https://www.eia.gov/opendata/register.php and set EIA_API_KEY."
            )
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._page_length = page_length
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def __enter__(self) -> EIAClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_demand(
        self,
        respondents: Iterable[str],
        start: datetime,
        end: datetime,
        data_types: Iterable[str] = ("D",),
    ) -> list[DemandRecord]:
        """Fetch every matching record in [start, end], following pagination."""
        records = list(
            self._iter_records(
                respondents=list(respondents),
                start=start,
                end=end,
                data_types=list(data_types),
            )
        )
        # Deterministic order keeps storage and tests stable.
        records.sort(key=lambda r: (r.respondent, r.data_type, r.period_utc))
        return records

    # -- internals ---------------------------------------------------------

    def _iter_records(
        self,
        respondents: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str],
    ) -> Iterator[DemandRecord]:
        offset = 0
        while True:
            payload = self._request_page(respondents, start, end, data_types, offset)
            response = payload.get("response", {})
            rows = response.get("data", []) or []
            for row in rows:
                yield self._row_to_record(row)

            fetched = offset + len(rows)
            total = int(response.get("total", fetched) or 0)
            logger.debug("EIA page: offset=%d rows=%d total=%d", offset, len(rows), total)
            if not rows or fetched >= total:
                break
            offset = fetched

    def _request_page(
        self,
        respondents: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str],
        offset: int,
    ) -> dict:
        params = self._build_params(respondents, start, end, data_types, offset)
        retryer = Retrying(
            stop=stop_after_attempt(max(1, self._max_retries)),
            wait=wait_exponential(min=self._backoff_min, max=self._backoff_max),
            retry=retry_if_exception_type(EIARetryableError),
            reraise=True,
        )
        return retryer(self._do_request, params)

    def _do_request(self, params: list[tuple[str, str]]) -> dict:
        try:
            resp = self._client.get(self._base_url, params=params)
        except httpx.TransportError as exc:  # timeouts, connection errors
            raise EIARetryableError(f"transport error contacting EIA: {exc}") from exc

        if resp.status_code == 429:
            raise EIARetryableError("EIA API rate limit (HTTP 429)")
        if resp.status_code >= 500:
            raise EIARetryableError(f"EIA API server error (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            raise EIAError(f"EIA API error (HTTP {resp.status_code}): {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise EIAError("EIA API returned a non-JSON response") from exc
        if "response" not in payload:
            # v2 surfaces failures under an "error" key instead of "response".
            raise EIAError(f"unexpected EIA payload: {str(payload)[:300]}")
        return payload

    def _build_params(
        self,
        respondents: list[str],
        start: datetime,
        end: datetime,
        data_types: list[str],
        offset: int,
    ) -> list[tuple[str, str]]:
        # A list of tuples is the reliable way to send EIA's repeated bracket
        # params (facets[respondent][]=A&facets[respondent][]=B).
        params: list[tuple[str, str]] = [
            ("api_key", self._api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("start", start.strftime(_EIA_PERIOD_FORMAT)),
            ("end", end.strftime(_EIA_PERIOD_FORMAT)),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", str(offset)),
            ("length", str(self._page_length)),
        ]
        params.extend(("facets[respondent][]", r) for r in respondents)
        params.extend(("facets[type][]", t) for t in data_types)
        return params

    def _row_to_record(self, row: dict) -> DemandRecord:
        try:
            period = parse_eia_period(row["period"])
        except (KeyError, ValueError) as exc:
            raise EIAError(f"could not parse period from row: {row}") from exc
        return DemandRecord(
            respondent=row.get("respondent", ""),
            respondent_name=row.get("respondent-name"),
            period_utc=period,
            data_type=row.get("type", ""),
            value=_coerce_value(row.get("value")),
            value_units=row.get("value-units"),
        )
