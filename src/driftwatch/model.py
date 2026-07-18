"""Train, persist, load, and serve predictions from the demand forecaster.

The model is deliberately not the hard part: a gradient-boosted regressor
(scikit-learn's HistGradientBoostingRegressor) on calendar + lag features, with
an honest time-ordered validation split and a seasonal-naive baseline to prove
the model is actually adding skill.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as F

logger = logging.getLogger(__name__)

MODEL_VERSION = 1
MIN_TRAINING_ROWS = 24 * 10  # ~10 days of complete feature rows


class NotEnoughDataError(RuntimeError):
    """Raised when there is too little history to train a useful model."""


@dataclass
class TrainingReport:
    respondent: str
    data_type: str
    model: str
    model_version: int
    trained_at_utc: str
    n_train: int
    n_val: int
    train_start_utc: str | None
    train_end_utc: str | None
    val_start_utc: str | None
    val_end_utc: str | None
    features: list[str]
    metrics: dict[str, float]


@dataclass
class Artifact:
    estimator: Any
    metadata: dict[str, Any]

    @property
    def trained_at_utc(self) -> str | None:
        return self.metadata.get("trained_at_utc")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _iso(ts) -> str | None:
    return None if ts is None else pd.Timestamp(ts).isoformat()


def train(
    frame: pd.DataFrame,
    *,
    respondent: str,
    data_type: str = "D",
    val_fraction: float = 0.2,
    random_state: int = 0,
) -> Artifact:
    """Fit the forecaster on ``frame`` and return an Artifact with metrics metadata."""
    X, y = F.training_matrix(frame)
    if len(X) < MIN_TRAINING_ROWS:
        raise NotEnoughDataError(
            f"need at least {MIN_TRAINING_ROWS} complete feature rows to train, got {len(X)}. "
            "Ingest more history (or seed synthetic data) first."
        )

    n_val = max(1, int(len(X) * val_fraction))
    split = len(X) - n_val
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    estimator = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_val)
    metrics = _metrics(y_val.to_numpy(), y_pred)
    # Seasonal-naive baseline: "same hour yesterday" (the lag_24 feature).
    baseline = _metrics(y_val.to_numpy(), X_val["lag_24"].to_numpy())
    metrics["baseline_mae"] = baseline["mae"]
    metrics["skill_vs_baseline"] = (
        float(1.0 - metrics["mae"] / baseline["mae"]) if baseline["mae"] else 0.0
    )

    report = TrainingReport(
        respondent=respondent,
        data_type=data_type,
        model=type(estimator).__name__,
        model_version=MODEL_VERSION,
        trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        n_train=len(X_train),
        n_val=len(X_val),
        train_start_utc=_iso(X_train.index.min()),
        train_end_utc=_iso(X_train.index.max()),
        val_start_utc=_iso(X_val.index.min()),
        val_end_utc=_iso(X_val.index.max()),
        features=F.FEATURE_COLUMNS,
        metrics={k: round(v, 4) for k, v in metrics.items()},
    )
    logger.info(
        "trained %s: %d rows, MAE=%.1f RMSE=%.1f MAPE=%.2f%% (baseline MAE=%.1f, skill=%.1f%%)",
        report.model,
        report.n_train,
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["baseline_mae"],
        metrics["skill_vs_baseline"] * 100,
    )
    return Artifact(estimator=estimator, metadata=asdict(report))


def save(artifact: Artifact, model_path: Path, meta_path: Path | None = None) -> None:
    meta_path = meta_path or model_path.with_suffix(".json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.estimator, model_path)
    meta_path.write_text(json.dumps(artifact.metadata, indent=2))


def load(model_path: Path, meta_path: Path | None = None) -> Artifact:
    meta_path = meta_path or model_path.with_suffix(".json")
    estimator = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return Artifact(estimator=estimator, metadata=metadata)


def try_load(model_path: Path, meta_path: Path | None = None) -> Artifact | None:
    meta_path = meta_path or model_path.with_suffix(".json")
    if not model_path.exists() or not meta_path.exists():
        return None
    try:
        return load(model_path, meta_path)
    except Exception as exc:  # corrupt/incompatible artifact — serve degrades gracefully
        logger.warning("could not load model artifact at %s: %s", model_path, exc)
        return None


def predict(artifact: Artifact, history: pd.DataFrame, periods) -> pd.Series:
    """Predict demand for ``periods``; entries are NaN where features are incomplete."""
    feats = F.inference_matrix(history, periods)
    preds = pd.Series(np.nan, index=feats.index, dtype="float64", name="predicted")
    usable = feats.dropna()
    if len(usable):
        preds.loc[usable.index] = artifact.estimator.predict(usable[F.FEATURE_COLUMNS])
    return preds
