"""Command-line entry point: `driftwatch ingest | status | init-db`."""

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
