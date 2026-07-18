"""Tests for training, persistence, and prediction."""

from __future__ import annotations

import pandas as pd
import pytest

from driftwatch import features as F
from driftwatch import model as M
from driftwatch.synthetic import synthetic_records


def _frame(days: int = 40, seed: int = 1) -> pd.DataFrame:
    records = synthetic_records(hours=days * 24, seed=seed)
    rows = [(r.period_utc.isoformat(), r.value) for r in records]
    return F.frame_from_observations(rows)


def test_train_produces_a_skillful_model():
    artifact = M.train(_frame(), respondent="PJM")
    metrics = artifact.metadata["metrics"]
    assert metrics["mae"] > 0
    assert metrics["mape"] < 12  # synthetic demand is learnable
    # The booster should beat the seasonal-naive (same-hour-yesterday) baseline.
    assert metrics["skill_vs_baseline"] > 0
    assert artifact.metadata["n_train"] > 0
    assert artifact.metadata["features"] == F.FEATURE_COLUMNS


def test_train_raises_when_history_too_short():
    with pytest.raises(M.NotEnoughDataError):
        M.train(_frame(days=5), respondent="PJM")


def test_save_load_roundtrip(tmp_path):
    artifact = M.train(_frame(), respondent="PJM")
    model_path = tmp_path / "m.joblib"
    M.save(artifact, model_path)
    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()

    loaded = M.load(model_path)
    assert loaded.metadata["metrics"] == artifact.metadata["metrics"]
    assert loaded.trained_at_utc == artifact.trained_at_utc


def test_predict_returns_positive_values_for_next_24h():
    frame = _frame()
    artifact = M.train(frame, respondent="PJM")
    last = frame.index.max()
    periods = [last + pd.Timedelta(hours=k) for k in range(1, 25)]

    preds = M.predict(artifact, frame, periods)
    assert len(preds) == 24
    assert preds.notna().all()
    assert (preds > 0).all()
