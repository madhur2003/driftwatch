"""Drift detection: input-distribution shift (PSI + KS) and prediction-error decay.

This is the self-monitoring layer — the part that turns "a model" into "a system
that knows when it is becoming untrustworthy". Against a reference captured at
training time, it scores a recent window of demand and raises a flag when either:

  * the inputs have moved  — Population Stability Index (PSI) and a
    Kolmogorov-Smirnov two-sample test on the demand distribution; or
  * the model has decayed  — recent prediction error climbs above the error the
    model achieved at training time.

Calendar features are deliberately not monitored for PSI: their distribution is
fixed by the calendar window, not by real-world drift, so PSI on them measures a
window artifact rather than a signal. Demand *level* is what actually drifts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import features as F
from . import model as M

logger = logging.getLogger(__name__)

# Columns whose distribution we track. Demand ("value") is the signal.
MONITORED_COLUMNS: tuple[str, ...] = ("value",)

PSI_BINS = 10
PSI_WARN = 0.10
PSI_ALERT = 0.25
MAE_RATIO_WARN = 1.5
MAE_RATIO_ALERT = 2.0
_EPS = 1e-6

OK, WARN, ALERT = "ok", "warn", "alert"
_RANK = {OK: 0, WARN: 1, ALERT: 2}


class DriftError(RuntimeError):
    """Raised when drift cannot be evaluated (no model, no reference, no data)."""


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _RANK.get(s, 0)) if statuses else OK


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _iso(ts) -> str | None:
    return None if ts is None else pd.Timestamp(ts).isoformat()


# ---- PSI -----------------------------------------------------------------


def quantile_edges(sample, bins: int = PSI_BINS) -> np.ndarray:
    """Bin edges at evenly spaced quantiles of ``sample`` (ties collapsed)."""
    arr = np.asarray(sample, dtype=float)
    arr = arr[~np.isnan(arr)]
    edges = np.quantile(arr, np.linspace(0.0, 1.0, bins + 1))
    return np.unique(edges)


def _bin_proportions(values, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    inner = np.asarray(edges, dtype=float)[1:-1]
    idx = np.digitize(arr, inner)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    total = counts.sum()
    if total == 0:
        return np.full(len(edges) - 1, np.nan)
    return counts / total


def psi_from_edges(edges, expected, current) -> float:
    """PSI of ``current`` against fixed reference ``edges``/``expected`` proportions."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 3:  # degenerate / near-constant reference
        return 0.0
    actual = _bin_proportions(current, edges)
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(actual, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def population_stability_index(reference, current, bins: int = PSI_BINS) -> float:
    """PSI computed directly from two samples (bins derived from ``reference``)."""
    edges = quantile_edges(reference, bins)
    if len(edges) < 3:
        return 0.0
    expected = _bin_proportions(reference, edges)
    return psi_from_edges(edges, expected, current)


def psi_status(psi: float) -> str:
    if np.isnan(psi):
        return OK
    if psi >= PSI_ALERT:
        return ALERT
    if psi >= PSI_WARN:
        return WARN
    return OK


def error_status(ratio: float | None) -> str:
    if ratio is None:
        return OK
    if ratio >= MAE_RATIO_ALERT:
        return ALERT
    if ratio >= MAE_RATIO_WARN:
        return WARN
    return OK


# ---- reference capture (at training time) --------------------------------


def build_reference(
    columns: dict[str, np.ndarray],
    *,
    bins: int = PSI_BINS,
    max_sample: int = 1000,
    seed: int = 0,
) -> dict:
    """Capture PSI bin edges/proportions and a KS sample for each monitored column."""
    rng = np.random.default_rng(seed)
    reference: dict[str, dict] = {}
    for name, raw in columns.items():
        arr = np.asarray(raw, dtype=float)
        arr = arr[~np.isnan(arr)]
        edges = quantile_edges(arr, bins)
        expected = _bin_proportions(arr, edges)
        sample = arr if arr.size <= max_sample else rng.choice(arr, max_sample, replace=False)
        reference[name] = {
            "edges": edges.tolist(),
            "expected": [None if np.isnan(x) else float(x) for x in expected],
            "sample": sample.tolist(),
            "n": int(arr.size),
        }
    return reference


# ---- report structures ---------------------------------------------------


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float | None
    ks_pvalue: float | None
    n_current: int
    status: str


@dataclass
class ErrorDrift:
    n: int
    mae: float | None
    mape: float | None
    baseline_mae: float | None
    mae_ratio: float | None
    status: str


@dataclass
class DriftReport:
    respondent: str
    data_type: str
    generated_at_utc: str
    current_start_utc: str | None
    current_end_utc: str | None
    n_current: int
    features: list[FeatureDrift]
    error: ErrorDrift
    status: str
    flagged: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def value_psi(self) -> float | None:
        for feat in self.features:
            if feat.feature == "value":
                return feat.psi
        return None


# ---- computation ---------------------------------------------------------


def _column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column in ("value", F.TARGET):
        series = frame["value"]
    else:
        series = F.build_features(frame)[column]
    return series.dropna().to_numpy(dtype=float)


def compute_feature_drift(
    reference: dict,
    current_frame: pd.DataFrame,
    columns: tuple[str, ...] = MONITORED_COLUMNS,
) -> list[FeatureDrift]:
    results: list[FeatureDrift] = []
    for column in columns:
        ref = reference.get(column)
        current = _column_values(current_frame, column)
        if ref is None or current.size == 0:
            results.append(FeatureDrift(column, float("nan"), None, None, int(current.size), OK))
            continue

        expected = [_EPS if x is None else x for x in ref["expected"]]
        psi = psi_from_edges(ref["edges"], expected, current)

        ks_stat = ks_p = None
        ref_sample = np.asarray(ref.get("sample", []), dtype=float)
        if ref_sample.size and current.size:
            result = ks_2samp(ref_sample, current)
            ks_stat, ks_p = float(result.statistic), float(result.pvalue)

        results.append(
            FeatureDrift(
                feature=column,
                psi=round(psi, 4),
                ks_statistic=round(ks_stat, 4) if ks_stat is not None else None,
                ks_pvalue=round(ks_p, 6) if ks_p is not None else None,
                n_current=int(current.size),
                status=psi_status(psi),
            )
        )
    return results


def compute_error_drift(
    artifact: M.Artifact,
    history: pd.DataFrame,
    periods,
    baseline_mae: float | None,
) -> ErrorDrift:
    periods = list(periods)
    if not periods:
        return ErrorDrift(0, None, None, baseline_mae, None, OK)

    preds = M.predict(artifact, history, periods)
    actual = history["value"].reindex(preds.index)
    paired = pd.DataFrame({"pred": preds, "actual": actual}).dropna()
    if paired.empty:
        return ErrorDrift(0, None, None, baseline_mae, None, OK)

    err = (paired["pred"] - paired["actual"]).to_numpy()
    mae = float(np.mean(np.abs(err)))
    denom = np.where(paired["actual"].to_numpy() == 0, np.nan, paired["actual"].to_numpy())
    mape = float(np.nanmean(np.abs(err / denom)) * 100)
    ratio = mae / baseline_mae if baseline_mae else None
    return ErrorDrift(
        n=len(paired),
        mae=round(mae, 2),
        mape=round(mape, 4),
        baseline_mae=round(baseline_mae, 2) if baseline_mae else None,
        mae_ratio=round(ratio, 4) if ratio is not None else None,
        status=error_status(ratio),
    )


def analyze(
    artifact: M.Artifact,
    *,
    respondent: str,
    data_type: str,
    current_frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
    error_periods=None,
) -> DriftReport:
    """Assemble a DriftReport from input-distribution drift and error drift."""
    reference = artifact.reference or {}
    baseline_mae = (artifact.metadata.get("metrics") or {}).get("mae")

    feature_drift = compute_feature_drift(reference, current_frame)
    if history is not None and error_periods is not None:
        error_drift = compute_error_drift(artifact, history, error_periods, baseline_mae)
    else:
        error_drift = ErrorDrift(0, None, None, baseline_mae, None, OK)

    overall = _worst(*(f.status for f in feature_drift), error_drift.status)
    index = current_frame.index
    return DriftReport(
        respondent=respondent,
        data_type=data_type,
        generated_at_utc=_now_iso(),
        current_start_utc=_iso(index.min()) if len(index) else None,
        current_end_utc=_iso(index.max()) if len(index) else None,
        n_current=int(current_frame["value"].notna().sum()) if len(index) else 0,
        features=feature_drift,
        error=error_drift,
        status=overall,
        flagged=overall != OK,
    )


def evaluate(
    settings,
    *,
    artifact: M.Artifact | None = None,
    respondent: str = "PJM",
    data_type: str = "D",
    window_hours: int = 168,
    error_window_hours: int = 168,
) -> DriftReport:
    """Load data + model and evaluate drift over the most recent window."""
    from . import db

    if artifact is None:
        artifact = M.try_load(settings.model_path, settings.meta_path, settings.reference_path)
    if artifact is None:
        raise DriftError("no trained model available; run `driftwatch train` first")
    if not artifact.reference:
        raise DriftError("model has no drift reference; retrain to capture one")

    with db.get_connection(settings.db_path) as conn:
        rows = db.select_observations(conn, respondent, data_type)
    frame = F.frame_from_observations(rows)
    if frame.empty:
        raise DriftError(f"no observations for respondent '{respondent}'")

    last = frame.index.max()
    current = frame[frame.index >= last - pd.Timedelta(hours=window_hours - 1)]
    error_start = last - pd.Timedelta(hours=error_window_hours - 1)
    error_periods = frame.index[(frame.index >= error_start) & frame["value"].notna()]
    return analyze(
        artifact,
        respondent=respondent,
        data_type=data_type,
        current_frame=current,
        history=frame,
        error_periods=list(error_periods),
    )


def _clean(x: float | None) -> float | None:
    """NaN -> None so SQLite stores a NULL rather than a non-portable NaN."""
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def store_report(conn, report: DriftReport) -> int:
    """Persist a DriftReport into the drift_reports table; returns its row id."""
    from . import db

    value_feat = next((f for f in report.features if f.feature == "value"), None)
    return db.insert_drift_report(
        conn,
        respondent=report.respondent,
        data_type=report.data_type,
        generated_at_utc=report.generated_at_utc,
        current_start=report.current_start_utc,
        current_end=report.current_end_utc,
        status=report.status,
        flagged=report.flagged,
        value_psi=_clean(value_feat.psi) if value_feat else None,
        ks_statistic=_clean(value_feat.ks_statistic) if value_feat else None,
        ks_pvalue=_clean(value_feat.ks_pvalue) if value_feat else None,
        error_mae=_clean(report.error.mae),
        baseline_mae=_clean(report.error.baseline_mae),
        mae_ratio=_clean(report.error.mae_ratio),
        report_json=json.dumps(report.to_dict()),
    )
