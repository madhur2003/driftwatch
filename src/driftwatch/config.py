"""Runtime configuration, resolved from environment variables (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present. In CI the variables are set directly, so this
# is a no-op there.
load_dotenv()

# Default store/model locations, relative to the current working directory — so
# the same paths resolve whether Driftwatch runs from source or an installed
# wheel. The container overrides both via environment variables.
DEFAULT_DB_PATH = Path("data/driftwatch.db")
DEFAULT_MODEL_PATH = Path("models/model.joblib")

# EIA API v2 — hourly electricity demand by balancing authority, reported in UTC.
# Docs: https://www.eia.gov/opendata/
EIA_API_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# Balancing authorities tracked by default. PJM is large and richly structured,
# which gives the drift layer (later weeks) something real to catch.
DEFAULT_RESPONDENTS: tuple[str, ...] = ("PJM",)

# EIA "type" codes on the region-data feed:
#   D = Demand, DF = Day-ahead demand forecast, NG = Net generation, TI = Total interchange.
DEFAULT_DATA_TYPES: tuple[str, ...] = ("D",)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    eia_api_key: str | None
    db_path: Path
    request_timeout: float
    max_retries: int
    model_path: Path = DEFAULT_MODEL_PATH

    @property
    def meta_path(self) -> Path:
        """Sidecar JSON path holding the trained model's metadata."""
        return self.model_path.with_suffix(".json")

    @property
    def reference_path(self) -> Path:
        """Sidecar JSON path holding the drift reference distribution."""
        return self.model_path.parent / (self.model_path.stem + ".reference.json")

    @classmethod
    def from_env(cls) -> Settings:
        db_path = os.environ.get("DRIFTWATCH_DB_PATH")
        model_path = os.environ.get("DRIFTWATCH_MODEL_PATH")
        return cls(
            eia_api_key=os.environ.get("EIA_API_KEY"),
            db_path=Path(db_path) if db_path else DEFAULT_DB_PATH,
            request_timeout=float(os.environ.get("DRIFTWATCH_HTTP_TIMEOUT", "30")),
            max_retries=int(os.environ.get("DRIFTWATCH_MAX_RETRIES", "4")),
            model_path=Path(model_path) if model_path else DEFAULT_MODEL_PATH,
        )
