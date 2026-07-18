"""Feature engineering for the hourly demand forecaster.

A single code path builds features for both training and inference, so there is
no train/serve skew. Every feature is derived from history strictly older than
the target hour (calendar terms, fixed lags of >= 24h, and a rolling mean of the
prior day), which keeps forecasts up to 24h ahead leakage-free and lets the API
build features for future hours purely from stored observations.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TARGET = "value"

FEATURE_COLUMNS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "lag_24",
    "lag_48",
    "lag_168",
    "roll_mean_24",
]

# Longest history any feature reaches back for (hours): the 168h lag plus the
# 24h averaging window that ends 24h before the target. Inference needs at least
# this much history before the first target hour for features to be complete.
MAX_LOOKBACK_HOURS = 168 + 24


def frame_from_observations(rows: Iterable[tuple[str, float | None]]) -> pd.DataFrame:
    """Build a UTC-hour-indexed frame with one ``value`` column from (period, value) pairs."""
    rows = list(rows)
    empty_index = pd.DatetimeIndex([], tz="UTC", name="period")
    if not rows:
        return pd.DataFrame({"value": pd.Series(dtype="float64")}, index=empty_index)
    periods, values = zip(*rows, strict=True)
    index = pd.DatetimeIndex(pd.to_datetime(list(periods), utc=True), name="period")
    out = pd.DataFrame({"value": pd.to_numeric(list(values), errors="coerce")}, index=index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _complete_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a gap-free hourly range so time-based shifts line up with real time."""
    if frame.empty:
        return frame
    full = pd.date_range(frame.index.min(), frame.index.max(), freq="h", tz="UTC", name="period")
    return frame.reindex(full)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix plus target for every hour of a (possibly gappy) series."""
    df = _complete_hourly(frame)
    if df.empty:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, TARGET])

    idx = df.index
    value = df["value"]
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    feats = pd.DataFrame(index=idx)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats["is_weekend"] = (dow >= 5).astype(float)
    feats["lag_24"] = value.shift(24)
    feats["lag_48"] = value.shift(48)
    feats["lag_168"] = value.shift(168)
    # Mean of the 24h window ending 24h before the target (the prior day),
    # so it is known for any target within 24h of the last observation.
    feats["roll_mean_24"] = value.shift(24).rolling(24).mean()
    feats[TARGET] = value
    return feats


def training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for rows with all features present and a known target."""
    feats = build_features(frame)
    complete = feats.dropna(subset=[*FEATURE_COLUMNS, TARGET])
    return complete[FEATURE_COLUMNS], complete[TARGET]


def inference_matrix(history: pd.DataFrame, periods: Iterable) -> pd.DataFrame:
    """Feature rows for the requested target periods, using ``history`` for lags.

    Target hours are added to the series with an unknown (NaN) value; their
    features still resolve because every feature reads only past history. Rows
    whose features are incomplete are returned with NaNs for the caller to skip.
    """
    target_index = pd.DatetimeIndex(pd.to_datetime(list(periods), utc=True), name="period")
    combined = history.copy()
    missing = target_index.difference(combined.index)
    if len(missing):
        combined = pd.concat([combined, pd.DataFrame({"value": np.nan}, index=missing)])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    feats = build_features(combined)
    return feats.reindex(target_index)[FEATURE_COLUMNS]
