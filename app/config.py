"""Runtime configuration, read from environment variables (set in .env / docker-compose / Azure Container Apps).

Change behaviour here or through the environment; nothing else in the code reads os.environ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model bundle and examples exported by the notebook (section 19b)
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/neighbour_impact_models.joblib"))
    examples_path: Path = Path(os.getenv("EXAMPLES_PATH", "models/deploy_test_cases.json"))
    # Data store exported by the notebook (section 19c): cells, neighbours, outages, KPI readings
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

    # Warning thresholds: a neighbour is "degraded" if the prediction is more than 20% above its pre-outage value
    degraded_threshold: float = float(os.getenv("DEGRADED_THRESHOLD", "0.20"))
    severe_threshold: float = float(os.getenv("SEVERE_THRESHOLD", "0.50"))

    # Security and limits
    api_key: str | None = os.getenv("API_KEY") or None          # secret; None = no authentication (local only)
    max_neighbours: int = int(os.getenv("MAX_NEIGHBOURS", "60"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "50"))
    allow_any_time: bool = _bool("ALLOW_ANY_TIME")                # outages-mode store: allow times that are not real outages

    # Operations
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    app_version: str = os.getenv("APP_VERSION", "local")         # set to the git commit SHA by CI/CD


settings = Settings()
