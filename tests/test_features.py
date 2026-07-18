"""Tests for feature engineering — especially that features never leak the future."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from driftwatch import features as F


def _ramp_series(n: int = 400) -> pd.DataFrame:
    """A gap-free hourly series whose value equals its integer hour offset."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [((start + timedelta(hours=i)).isoformat(), float(i)) for i in range(n)]
    return F.frame_from_observations(rows)


def test_frame_from_observations_is_sorted_hourly():
    frame = _ramp_series(5)
    assert list(frame.columns) == ["value"]
    assert frame.index.tz is not None
    assert frame["value"].tolist() == [0, 1, 2, 3, 4]


def test_frame_from_observations_empty():
    frame = F.frame_from_observations([])
    assert frame.empty
    assert "value" in frame.columns


def test_lag_features_reference_only_the_past():
    feats = F.build_features(_ramp_series(400))
    row = feats.iloc[200]
    assert row["lag_24"] == 200 - 24
    assert row["lag_48"] == 200 - 48
    assert row["lag_168"] == 200 - 168
    # roll_mean_24 = mean of the 24h window ending 24h before t -> value[153..176]
    assert row["roll_mean_24"] == pytest.approx((153 + 176) / 2)


def test_training_matrix_drops_warmup_and_has_no_nans():
    X, y = F.training_matrix(_ramp_series(400))
    # lag_168 is the binding constraint: the first 168 rows lack it.
    assert len(X) == 400 - 168
    assert list(X.columns) == F.FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert len(y) == len(X)


def test_inference_matrix_builds_complete_features_for_future_hours():
    history = _ramp_series(400)  # values 0..399, last hour = offset 399
    last = history.index.max()
    future = [last + pd.Timedelta(hours=k) for k in (1, 2, 24)]
    Xf = F.inference_matrix(history, future)

    assert list(Xf.index) == future
    # +1h target (offset 400): lag_24 must be value[376].
    assert Xf.iloc[0]["lag_24"] == 376
    # All three future hours (<= 24h ahead) must be fully computable from history.
    assert not Xf[F.FEATURE_COLUMNS].isna().any().any()
