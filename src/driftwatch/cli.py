"""Command-line entry point.

Subcommands: `ingest | status | init-db | synth | train | predict | serve | drift |
bootstrap`.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from . import db
from .config import DEFAULT_DATA_TYPES, DEFAULT_RESPONDENTS, Settings
from .eia_client import EIAError
from .ingest import DEFAULT_LOOKBACK_HOURS, run_ingestion

logger = logging.getLogger("driftwatch")


def _parse_dt(value: str) -> datetime:
    """Accept YYYY-MM-DD or YYYY-MM-DDTHH, interpreted as UTC."""
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid datetime '{value}'; use YYYY-MM-DD or YYYY-MM-DDTHH (UTC)"
    )


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    result = run_ingestion(
        settings,
        respondents=args.respondent or list(DEFAULT_RESPONDENTS),
        data_types=args.type or list(DEFAULT_DATA_TYPES),
        start=args.start,
        end=args.end,
        lookback_hours=args.lookback_hours,
    )
    print(
        f"Fetched {result.records_fetched} records, wrote {result.rows_upserted} rows "
        f"for {','.join(result.respondents)} "
        f"[{result.window_start.isoformat()} .. {result.window_end.isoformat()}]"
    )
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    with db.get_connection(settings.db_path) as conn:
        total = db.observation_count(conn)
        print(f"Database:      {settings.db_path}")
        print(f"Observations:  {total}")
        for respondent in args.respondent or list(DEFAULT_RESPONDENTS):
            latest = db.latest_period(conn, respondent)
            print(f"  {respondent:<8} latest demand period: {latest or '(none)'}")
        runs = db.recent_runs(conn, limit=args.limit)
        if runs:
            print("Recent runs:")
            for run in runs:
                print(
                    f"  #{run['id']:<4} {run['status']:<8} "
                    f"rows={run['rows_upserted'] or 0:<6} "
                    f"{run['started_at_utc']} "
                    f"{'ERROR: ' + run['error'] if run['error'] else ''}"
                )
    return 0


def cmd_init_db(_args: argparse.Namespace, settings: Settings) -> int:
    with db.get_connection(settings.db_path):
        pass
    print(f"Initialized database at {settings.db_path}")
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    # Heavy ML imports are deferred so `ingest`/`status` stay lightweight.
    from . import features as ft
    from . import model as ml

    respondent = (args.respondent or list(DEFAULT_RESPONDENTS))[0]
    data_type = (args.type or list(DEFAULT_DATA_TYPES))[0]
    with db.get_connection(settings.db_path) as conn:
        rows = db.select_observations(conn, respondent, data_type)
    frame = ft.frame_from_observations(rows)
    try:
        artifact = ml.train(
            frame,
            respondent=respondent,
            data_type=data_type,
            val_fraction=args.val_fraction,
        )
    except ml.NotEnoughDataError as exc:
        logger.error("%s", exc)
        return 1
    ml.save(artifact, settings.model_path, settings.meta_path)

    meta = artifact.metadata
    m = meta["metrics"]
    print(
        f"Trained {meta['model']} for {respondent} (train={meta['n_train']}, val={meta['n_val']})"
    )
    print(
        f"  val MAE={m['mae']:,.1f}  RMSE={m['rmse']:,.1f}  MAPE={m['mape']}%  "
        f"skill vs seasonal-naive baseline={m['skill_vs_baseline'] * 100:.1f}%"
    )
    print(f"  saved -> {settings.model_path}")
    return 0


def cmd_predict(args: argparse.Namespace, settings: Settings) -> int:
    from fastapi import HTTPException

    from . import model as ml
    from .api import PredictRequest, _predict  # reuse the exact serving path

    respondent = (args.respondent or list(DEFAULT_RESPONDENTS))[0]
    data_type = (args.type or list(DEFAULT_DATA_TYPES))[0]
    artifact = ml.try_load(settings.model_path, settings.meta_path)
    if artifact is None:
        logger.error("no trained model at %s; run `driftwatch train` first", settings.model_path)
        return 1

    req = PredictRequest(respondent=respondent, data_type=data_type, horizon_hours=args.horizon)
    try:
        resp = _predict(settings, artifact, req)
    except HTTPException as exc:
        logger.error("%s", exc.detail)
        return 1

    print(
        f"Forecast for {resp.respondent} ({resp.data_type}) — model trained {resp.model_trained_at}"
    )
    for item in resp.predictions:
        value = f"{item.predicted:,.0f} MWh" if item.predicted is not None else "(n/a)"
        print(f"  {item.period.isoformat()}  {value}")
    return 0


def cmd_synth(args: argparse.Namespace, settings: Settings) -> int:
    from .synthetic import synthetic_records

    respondent = (args.respondent or list(DEFAULT_RESPONDENTS))[0]
    records = synthetic_records(
        respondent=respondent,
        hours=args.days * 24,
        seed=args.seed,
        demand_multiplier=1.0 + args.shift,
    )
    with db.get_connection(settings.db_path) as conn:
        written = db.upsert_records(conn, records)
    shift_note = f", +{args.shift:.0%} level shift" if args.shift else ""
    print(
        f"Seeded {written} synthetic hourly rows for {respondent} "
        f"({args.days} days{shift_note}) into {settings.db_path}"
    )
    return 0


def cmd_serve(args: argparse.Namespace, _settings: Settings) -> int:
    import uvicorn

    uvicorn.run("driftwatch.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_drift(args: argparse.Namespace, settings: Settings) -> int:
    from . import drift as dr

    respondent = (args.respondent or list(DEFAULT_RESPONDENTS))[0]
    data_type = (args.type or list(DEFAULT_DATA_TYPES))[0]
    try:
        report = dr.evaluate(
            settings,
            respondent=respondent,
            data_type=data_type,
            window_hours=args.window_hours,
            error_window_hours=args.error_window_hours,
        )
    except dr.DriftError as exc:
        logger.error("%s", exc)
        return 1
    with db.get_connection(settings.db_path) as conn:
        dr.store_report(conn, report)

    print(
        f"Drift status: {report.status.upper()}  "
        f"({report.respondent} {report.data_type}, n={report.n_current}, "
        f"window {report.current_start_utc} .. {report.current_end_utc})"
    )
    for feat in report.features:
        psi_txt = "n/a" if feat.psi != feat.psi else f"{feat.psi:.4f}"  # NaN-safe
        ks_txt = (
            "n/a" if feat.ks_statistic is None else f"{feat.ks_statistic:.4f} (p={feat.ks_pvalue})"
        )
        print(f"  input '{feat.feature}': PSI={psi_txt} [{feat.status}]  KS={ks_txt}")
    err = report.error
    if err.mae is not None:
        print(
            f"  error: recent MAE={err.mae:,.1f} vs baseline {err.baseline_mae:,.1f} "
            f"(ratio {err.mae_ratio}) [{err.status}]  n={err.n}"
        )
    else:
        print(f"  error: (insufficient recent actuals) [{err.status}]")

    if args.fail_on_alert and report.status == dr.ALERT:
        return 2
    return 0


def cmd_bootstrap(args: argparse.Namespace, settings: Settings) -> int:
    """Provision a fresh deployment: seed demo data if sparse, then ensure a model."""
    from . import features as ft
    from . import model as ml
    from .synthetic import synthetic_records

    respondent = (args.respondent or list(DEFAULT_RESPONDENTS))[0]
    data_type = (args.type or list(DEFAULT_DATA_TYPES))[0]

    with db.get_connection(settings.db_path) as conn:
        existing = db.observation_count(conn)
        seeded = 0
        if args.demo and existing < args.min_observations:
            records = synthetic_records(respondent=respondent, hours=args.days * 24, seed=args.seed)
            seeded = db.upsert_records(conn, records)
        rows = db.select_observations(conn, respondent, data_type)

    print(
        f"bootstrap: {existing} existing observations"
        + (f", seeded {seeded} synthetic rows" if seeded else "")
    )

    if args.retrain or not settings.model_path.exists():
        frame = ft.frame_from_observations(rows)
        try:
            artifact = ml.train(frame, respondent=respondent, data_type=data_type)
        except ml.NotEnoughDataError as exc:
            logger.warning("bootstrap: skipping training — %s", exc)
            return 0
        ml.save(artifact, settings.model_path, settings.meta_path, settings.reference_path)
        print(f"bootstrap: trained model -> {settings.model_path}")
    else:
        print(f"bootstrap: model already present at {settings.model_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch",
        description="Ingest and monitor hourly electricity demand from the EIA API.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    # A shared parent so --log-level is also accepted *after* the subcommand
    # (e.g. `driftwatch ingest --log-level DEBUG`). SUPPRESS keeps the default
    # from clobbering a value already parsed before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Pull demand and store it.", parents=[common])
    p_ingest.add_argument(
        "--respondent",
        action="append",
        help="Balancing authority code, e.g. PJM (repeatable). Default: PJM.",
    )
    p_ingest.add_argument(
        "--type",
        action="append",
        help="EIA data type: D, DF, NG, TI (repeatable). Default: D.",
    )
    p_ingest.add_argument("--start", type=_parse_dt, help="Window start (UTC).")
    p_ingest.add_argument("--end", type=_parse_dt, help="Window end (UTC).")
    p_ingest.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help=f"Rolling window size when --start is omitted. Default: {DEFAULT_LOOKBACK_HOURS}.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="Show what has been ingested.", parents=[common])
    p_status.add_argument("--respondent", action="append", help="Filter to these codes.")
    p_status.add_argument("--limit", type=int, default=5, help="Recent runs to show.")
    p_status.set_defaults(func=cmd_status)

    p_init = sub.add_parser("init-db", help="Create the database schema.", parents=[common])
    p_init.set_defaults(func=cmd_init_db)

    p_train = sub.add_parser("train", help="Train the demand forecaster.", parents=[common])
    p_train.add_argument("--respondent", action="append", help="Balancing authority. Default: PJM.")
    p_train.add_argument("--type", action="append", help="EIA data type. Default: D.")
    p_train.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the most recent data held out for validation. Default: 0.2.",
    )
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Forecast the next N hours.", parents=[common])
    p_predict.add_argument(
        "--respondent", action="append", help="Balancing authority. Default: PJM."
    )
    p_predict.add_argument("--type", action="append", help="EIA data type. Default: D.")
    p_predict.add_argument(
        "--horizon", type=int, default=24, help="Hours ahead to forecast (1-24). Default: 24."
    )
    p_predict.set_defaults(func=cmd_predict)

    p_synth = sub.add_parser(
        "synth",
        help="Seed a synthetic demand series (offline demo / no API key needed).",
        parents=[common],
    )
    p_synth.add_argument("--respondent", action="append", help="Balancing authority. Default: PJM.")
    p_synth.add_argument(
        "--days", type=int, default=60, help="Days of history to generate. Default: 60."
    )
    p_synth.add_argument("--seed", type=int, default=0, help="RNG seed. Default: 0.")
    p_synth.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Fractional demand level shift to inject drift, e.g. 0.3 for +30%%. Default: 0.",
    )
    p_synth.set_defaults(func=cmd_synth)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service (uvicorn).", parents=[common])
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1.")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port. Default: 8000.")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    p_serve.set_defaults(func=cmd_serve)

    p_drift = sub.add_parser(
        "drift", help="Check for input drift and prediction-error decay.", parents=[common]
    )
    p_drift.add_argument("--respondent", action="append", help="Balancing authority. Default: PJM.")
    p_drift.add_argument("--type", action="append", help="EIA data type. Default: D.")
    p_drift.add_argument(
        "--window-hours", type=int, default=168, help="Recent window to score. Default: 168 (7d)."
    )
    p_drift.add_argument(
        "--error-window-hours",
        type=int,
        default=168,
        help="Window for prediction-error monitoring. Default: 168 (7d).",
    )
    p_drift.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 2 when the status is ALERT (for cron/CI alerting).",
    )
    p_drift.set_defaults(func=cmd_drift)

    p_boot = sub.add_parser(
        "bootstrap",
        help="Provision a deployment: seed demo data and train a model if missing.",
        parents=[common],
    )
    p_boot.add_argument("--respondent", action="append", help="Balancing authority. Default: PJM.")
    p_boot.add_argument("--type", action="append", help="EIA data type. Default: D.")
    p_boot.add_argument(
        "--demo",
        action="store_true",
        help="Seed a synthetic series when the store is sparse (no API key needed).",
    )
    p_boot.add_argument(
        "--days", type=int, default=45, help="Days of demo data to seed. Default: 45."
    )
    p_boot.add_argument("--seed", type=int, default=0, help="RNG seed for demo data. Default: 0.")
    p_boot.add_argument(
        "--min-observations",
        type=int,
        default=24 * 14,
        help="Only seed demo data below this many observations. Default: 336.",
    )
    p_boot.add_argument("--retrain", action="store_true", help="Retrain even if a model exists.")
    p_boot.set_defaults(func=cmd_bootstrap)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    settings = Settings.from_env()
    try:
        return int(args.func(args, settings))
    except EIAError as exc:
        # Expected, user-facing failures (bad/missing key, API errors): report
        # cleanly instead of dumping a traceback. The run is already logged as
        # failed by run_ingestion.
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
