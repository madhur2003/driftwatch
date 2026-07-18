"""Tests for the drift layer — PSI/KS math, thresholds, and that drift actually fires."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from driftwatch import db
from driftwatch import drift as D
from driftwatch import features as F
from driftwatch import model as M
from driftwatch.config import Settings
from driftwatch.synthetic import synthetic_records

# ---- PSI / status unit tests --------------------------------------------


def test_psi_is_small_for_identical_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 5000)
    cur = rng.normal(100, 10, 5000)
    assert D.population_stability_index(ref, cur) < D.PSI_WARN


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 5000)
    cur = rng.normal(140, 10, 5000)
    psi = D.population_stability_index(ref, cur)
    assert psi > D.PSI_ALERT
    assert D.psi_status(psi) == D.ALERT


def test_psi_and_error_status_thresholds():
    assert D.psi_status(0.05) == D.OK
    assert D.psi_status(0.15) == D.WARN
    assert D.psi_status(0.30) == D.ALERT
    assert D.error_status(None) == D.OK
    assert D.error_status(1.2) == D.OK
    assert D.error_status(1.6) == D.WARN
    assert D.error_status(2.5) == D.ALERT


def test_build_reference_shape():
    rng = np.random.default_rng(0)
    reference = D.build_reference({"value": rng.normal(100, 10, 500)})
    ref = reference["value"]
    assert len(ref["expected"]) == len(ref["edges"]) - 1
    assert ref["n"] == 500
    assert len(ref["sample"]) <= 1000


# ---- feature-drift detection --------------------------------------------


def _frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC", name="period")
    return pd.DataFrame({"value": values}, index=index)


def test_feature_drift_ok_when_stable_alert_when_shifted():
    rng = np.random.default_rng(1)
    reference = D.build_reference({"value": rng.normal(100, 10, 3000)})

    stable = D.compute_feature_drift(reference, _frame(rng.normal(100, 10, 400)))[0]
    assert stable.status == D.OK

    shifted = D.compute_feature_drift(reference, _frame(rng.normal(140, 10, 400)))[0]
    assert shifted.status == D.ALERT
    assert shifted.ks_pvalue is not None and shifted.ks_pvalue < 0.05


# ---- end-to-end via evaluate() ------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings(
        eia_api_key=None,
        db_path=tmp_path / "d.db",
        request_timeout=5.0,
        max_retries=1,
        model_path=tmp_path / "m.joblib",
    )


def _seed_and_train(settings, *, end, days=45, shift=0.0, seed=3):
    records = synthetic_records(hours=days * 24, end=end, seed=seed, demand_multiplier=1.0 + shift)
    with db.get_connection(settings.db_path) as conn:
        db.upsert_records(conn, records)
        rows = db.select_observations(conn, "PJM", "D")
    artifact = M.train(F.frame_from_observations(rows), respondent="PJM")
    M.save(artifact, settings.model_path, settings.meta_path, settings.reference_path)


def test_evaluate_stable_data_does_not_false_alarm(tmp_path):
    settings = _settings(tmp_path)
    _seed_and_train(settings, end=datetime(2026, 6, 1, tzinfo=UTC))

    report = D.evaluate(settings, respondent="PJM")
    assert report.status != D.ALERT
    assert report.value_psi() < D.PSI_ALERT


def test_evaluate_flags_alert_when_recent_window_is_shifted(tmp_path):
    settings = _settings(tmp_path)
    t0 = datetime(2026, 6, 1, tzinfo=UTC)
    _seed_and_train(settings, end=t0)  # reference captured on stable demand

    # A week of demand arrives ~35% higher than anything the model trained on.
    shifted = synthetic_records(
        hours=7 * 24, end=t0 + timedelta(days=7), seed=3, demand_multiplier=1.35
    )
    with db.get_connection(settings.db_path) as conn:
        db.upsert_records(conn, shifted)

    report = D.evaluate(settings, respondent="PJM", window_hours=168, error_window_hours=168)
    assert report.flagged
    assert report.status == D.ALERT
    assert report.value_psi() > D.PSI_ALERT

    # And it persists to the drift_reports table.
    with db.get_connection(settings.db_path) as conn:
        D.store_report(conn, report)
        latest = db.recent_drift_reports(conn, limit=1)[0]
    assert latest["status"] == "alert"
    assert latest["flagged"] == 1


def test_evaluate_without_model_raises(tmp_path):
    settings = _settings(tmp_path)
    with db.get_connection(settings.db_path) as conn:  # empty db, no model
        db.init_db(conn)
    with pytest.raises(D.DriftError):
        D.evaluate(settings, respondent="PJM")
