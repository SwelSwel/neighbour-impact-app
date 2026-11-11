"""Loads the trained model bundle once and turns validated inputs into predictions."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import features as F

log = logging.getLogger("predictor")
GRID_STEP_H = 3


def prediction_slot(outage_start: datetime) -> datetime:
    """First scheduled 3-hourly KPI reading at or after the outage start (the slot the model predicts).

    Matches the notebook: the slot index is ceil((start - midnight) / 3 h)."""
    midnight = outage_start.replace(hour=0, minute=0, second=0, microsecond=0)
    steps = math.ceil((outage_start - midnight).total_seconds() / (GRID_STEP_H * 3600))
    return midnight + timedelta(hours=GRID_STEP_H * steps)


class Predictor:
    def __init__(self, model_path: Path):
        self.bundle = joblib.load(model_path)
        b = self.bundle
        self.features: list[str] = b["features"]
        self.targets: list[str] = list(b["targets"])
        self.vars9 = b.get("vars9", F.VARS9)
        self.normal_vars = b.get("normal_vars", F.NORMAL_VARS)
        self.fail_vars = b.get("fail_vars", F.FAIL_VARS)
        trained = b.get("sklearn_version")
        if trained and trained != sklearn.__version__:
            log.warning("model trained with scikit-learn %s but running %s; predictions may differ",
                        trained, sklearn.__version__)
        log.info("model bundle loaded: %s, targets=%s, %d features", model_path, self.targets, len(self.features))

    def info(self) -> dict:
        b = self.bundle
        return {
            "model": b.get("deploy_model"), "targets": self.targets, "n_features": len(self.features),
            "trained_until": b.get("train_period_end"), "sklearn_trained": b.get("sklearn_version"),
            "sklearn_running": sklearn.__version__, "python_trained": b.get("python_version"),
            "test_metrics": {t: b["targets"][t].get("test_metrics") for t in self.targets},
        }

    def build_row(self, **kwargs) -> list[float]:
        f = F.impact_features(**kwargs, vars9=self.vars9, normal_vars=self.normal_vars, fail_vars=self.fail_vars)
        return [f[k] for k in self.features]

    def predict_rows(self, rows: list[list[float]]) -> dict[str, np.ndarray]:
        X = np.asarray(rows, dtype=np.float32)     # float32, exactly as in training
        return {t: self.bundle["targets"][t]["model"].predict(X) for t in self.targets}

    def interval(self, target: str) -> tuple[float, float]:
        lo, hi = self.bundle["targets"][target]["interval_80"]
        return float(lo), float(hi)
