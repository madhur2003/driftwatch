"""FastAPI service that serves the trained demand forecaster.

Built via a ``create_app(settings)`` factory so tests can point it at a
temporary database and model artifact. The module-level ``app`` is what
uvicorn / the container run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__, db
from . import drift as D
from . import features as F
from . import model as M
from .config import Settings

logger = logging.getLogger(__name__)


def _floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


# ---- request / response schemas -----------------------------------------


class PredictRequest(BaseModel):
    respondent: str = "PJM"
    data_type: str = "D"
    horizon_hours: int = Field(24, ge=1, le=24, description="Hours ahead to forecast.")
    periods: list[datetime] | None = Field(
        default=None,
        description="Explicit UTC hours to forecast; overrides horizon_hours when set.",
    )


class PredictionItem(BaseModel):
    period: datetime
    predicted: float | None


class PredictResponse(BaseModel):
    respondent: str
    data_type: str
    model_trained_at: str | None
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_trained_at: str | None


class DriftRequest(BaseModel):
    respondent: str = "PJM"
    data_type: str = "D"
    window_hours: int = Field(168, ge=24, le=24 * 90, description="Recent window to score.")
    error_window_hours: int = Field(
        168, ge=24, le=24 * 90, description="Window for prediction-error monitoring."
    )


# ---- prediction plumbing -------------------------------------------------


def _resolve_periods(settings: Settings, req: PredictRequest) -> list[datetime]:
    if req.periods:
        return sorted({_floor_hour(p) for p in req.periods})
    with db.get_connection(settings.db_path) as conn:
        latest = db.latest_period(conn, req.respondent, req.data_type)
    if latest is None:
        raise HTTPException(
            status_code=422,
            detail=f"no observations for respondent '{req.respondent}'; ingest or seed data first",
        )
    last = _floor_hour(pd.Timestamp(latest).to_pydatetime())
    return [last + timedelta(hours=i) for i in range(1, req.horizon_hours + 1)]


def _predict(settings: Settings, artifact: M.Artifact, req: PredictRequest) -> PredictResponse:
    periods = _resolve_periods(settings, req)
    window_start = (min(periods) - timedelta(hours=F.MAX_LOOKBACK_HOURS + 1)).isoformat()
    window_end = max(periods).isoformat()
    with db.get_connection(settings.db_path) as conn:
        rows: Iterable = db.select_observations(
            conn, req.respondent, req.data_type, start=window_start, end=window_end
        )
    history = F.frame_from_observations(rows)
    preds = M.predict(artifact, history, periods)

    items = [
        PredictionItem(
            period=pd.Timestamp(idx).to_pydatetime(),
            predicted=(None if pd.isna(value) else round(float(value), 1)),
        )
        for idx, value in preds.items()
    ]
    return PredictResponse(
        respondent=req.respondent,
        data_type=req.data_type,
        model_trained_at=artifact.trained_at_utc,
        predictions=items,
    )


# ---- app factory ---------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.artifact = M.try_load(
            settings.model_path, settings.meta_path, settings.reference_path
        )
        if app.state.artifact is None:
            logger.warning(
                "no model artifact at %s; /predict will 503 until one is trained",
                settings.model_path,
            )
        yield

    app = FastAPI(
        title="Driftwatch",
        version=__version__,
        summary="Hourly electricity-demand forecasting service.",
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict:
        return {"service": "driftwatch", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        artifact: M.Artifact | None = request.app.state.artifact
        return HealthResponse(
            status="ok",
            model_loaded=artifact is not None,
            model_trained_at=artifact.trained_at_utc if artifact else None,
        )

    @app.get("/model")
    def model_metadata(request: Request) -> dict:
        artifact: M.Artifact | None = request.app.state.artifact
        if artifact is None:
            raise HTTPException(status_code=404, detail="no trained model available")
        return artifact.metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request) -> PredictResponse:
        artifact: M.Artifact | None = request.app.state.artifact
        if artifact is None:
            raise HTTPException(
                status_code=503, detail="no trained model available; train one first"
            )
        return _predict(request.app.state.settings, artifact, req)

    @app.post("/drift")
    def drift(req: DriftRequest, request: Request) -> dict:
        settings = request.app.state.settings
        artifact: M.Artifact | None = request.app.state.artifact
        try:
            report = D.evaluate(
                settings,
                artifact=artifact,
                respondent=req.respondent,
                data_type=req.data_type,
                window_hours=req.window_hours,
                error_window_hours=req.error_window_hours,
            )
        except D.DriftError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        with db.get_connection(settings.db_path) as conn:
            D.store_report(conn, report)
        return report.to_dict()

    @app.get("/drift/history")
    def drift_history(request: Request, respondent: str | None = None, limit: int = 20) -> dict:
        settings = request.app.state.settings
        with db.get_connection(settings.db_path) as conn:
            rows = db.recent_drift_reports(conn, respondent=respondent, limit=limit)
        return {
            "reports": [
                {
                    "id": r["id"],
                    "generated_at_utc": r["generated_at_utc"],
                    "respondent": r["respondent"],
                    "status": r["status"],
                    "flagged": bool(r["flagged"]),
                    "value_psi": r["value_psi"],
                    "ks_statistic": r["ks_statistic"],
                    "error_mae": r["error_mae"],
                    "baseline_mae": r["baseline_mae"],
                    "mae_ratio": r["mae_ratio"],
                }
                for r in rows
            ]
        }

    return app


app = create_app()
