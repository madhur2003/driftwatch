"""Tests for the FastAPI service, driven through the app factory + TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from driftwatch import db
from driftwatch import features as F
from driftwatch import model as M
from driftwatch.api import create_app
from driftwatch.config import Settings
from driftwatch.synthetic import synthetic_records


def _settings(tmp_path, *, model_path=None) -> Settings:
    return Settings(
        eia_api_key=None,
        db_path=tmp_path / "d.db",
        request_timeout=5.0,
        max_retries=1,
        model_path=model_path or (tmp_path / "m.joblib"),
    )


@pytest.fixture
def trained_settings(tmp_path) -> Settings:
    settings = _settings(tmp_path)
    records = synthetic_records(hours=40 * 24, seed=2)
    with db.get_connection(settings.db_path) as conn:
        db.upsert_records(conn, records)
        rows = db.select_observations(conn, "PJM", "D")
    frame = F.frame_from_observations(rows)
    artifact = M.train(frame, respondent="PJM")
    M.save(artifact, settings.model_path, settings.meta_path)
    return settings


def test_health_reports_model_loaded(trained_settings):
    with TestClient(create_app(trained_settings)) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_trained_at"]


def test_model_endpoint_returns_metadata(trained_settings):
    with TestClient(create_app(trained_settings)) as client:
        resp = client.get("/model")
        assert resp.status_code == 200
        meta = resp.json()
        assert "metrics" in meta
        assert meta["respondent"] == "PJM"


def test_predict_horizon_returns_forecasts(trained_settings):
    with TestClient(create_app(trained_settings)) as client:
        resp = client.post("/predict", json={"respondent": "PJM", "horizon_hours": 6})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 6
        assert all(p["predicted"] and p["predicted"] > 0 for p in body["predictions"])


def test_predict_without_model_returns_503(tmp_path):
    settings = _settings(tmp_path, model_path=tmp_path / "missing.joblib")
    with TestClient(create_app(settings)) as client:
        resp = client.post("/predict", json={"respondent": "PJM"})
        assert resp.status_code == 503


def test_predict_unknown_respondent_returns_422(trained_settings):
    with TestClient(create_app(trained_settings)) as client:
        resp = client.post("/predict", json={"respondent": "NOPE", "horizon_hours": 3})
        assert resp.status_code == 422


def test_drift_endpoint_reports_status_and_records_history(trained_settings):
    with TestClient(create_app(trained_settings)) as client:
        resp = client.post(
            "/drift", json={"respondent": "PJM", "window_hours": 168, "error_window_hours": 168}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "warn", "alert")
        assert body["features"][0]["feature"] == "value"

        history = client.get("/drift/history")
        assert history.status_code == 200
        assert len(history.json()["reports"]) >= 1


def test_drift_without_model_returns_503(tmp_path):
    settings = _settings(tmp_path, model_path=tmp_path / "missing.joblib")
    with TestClient(create_app(settings)) as client:
        resp = client.post("/drift", json={"respondent": "PJM"})
        assert resp.status_code == 503
