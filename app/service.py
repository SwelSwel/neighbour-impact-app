"""Prediction logic shared by every endpoint: request -> 66 features -> three models -> warnings."""
from __future__ import annotations

import math

import numpy as np

from .config import settings
from .predictor import Predictor, prediction_slot
from .schemas import KPIForecast, NeighbourPrediction, PredictRequest, PredictResponse


def _nan(d: dict | None) -> dict:
    """None -> NaN, so a missing value becomes a missing feature exactly as in the notebook."""
    return {k: (np.nan if v is None else float(v)) for k, v in (d or {}).items()}


def _clean(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def forecast(before: float | None, pred: float, interval: tuple[float, float], actual: float | None = None) -> KPIForecast:
    has_before = before is not None and before > 0
    return KPIForecast(
        before=_clean(before), predicted=round(float(pred), 4),
        range_80_low=round(max(0.0, pred + interval[0]), 4), range_80_high=round(pred + interval[1], 4),
        change_pct=round(100 * (pred / before - 1), 2) if has_before else None,
        degraded=bool(has_before and pred > before * (1 + settings.degraded_threshold)),
        severe=bool(has_before and pred > before * (1 + settings.severe_threshold)),
        actual=_clean(actual))


def predict(p: Predictor, req: PredictRequest, request_id: str,
            actual: dict[str, dict | None] | None = None) -> PredictResponse:
    """`actual` (optional): observed values per neighbour, shown next to the predictions."""
    slot = prediction_slot(req.outage_start)
    failed = _nan(req.failed_cell.model_dump(exclude={"cell_id", "site_cells_down"}))
    nows = [_nan(n.now.model_dump()) for n in req.neighbours]
    total_cap = float(np.nansum([n["Capacity"] for n in nows]))

    rows = []
    for n, now in zip(req.neighbours, nows):
        rows.append(p.build_row(
            nb_now=now, nb_3h=_nan(n.h3_before.model_dump() if n.h3_before else None),
            nb_6h=_nan(n.h6_before.model_dump() if n.h6_before else None),
            failed=failed, n_absorbing=len(req.neighbours), total_absorbing_capacity=total_cap, hour=slot.hour,
            is_same_site=int(n.is_same_site), is_listed=int(n.is_listed),
            site_cells_down=req.failed_cell.site_cells_down,
            nb_last_week={k: v for k, v in n.last_week.model_dump().items() if v is not None} if n.last_week else None))
    preds = p.predict_rows(rows)

    out = []
    for i, (n, now) in enumerate(zip(req.neighbours, nows)):
        util = 100 * now["Load"] / now["Capacity"] if now["Capacity"] > 0 else np.nan
        before = {"Latency": now["Latency"], "Utilization": util, "Packet_Loss": now["Packet_Loss"]}
        act = (actual or {}).get(n.cell_id) or {}
        fc = {t: forecast(_clean(before[t]), preds[t][i], p.interval(t), act.get(t)) for t in p.targets}
        ratio = failed["Load"] / now["Capacity"] if now["Capacity"] > 0 else np.nan
        out.append(NeighbourPrediction(cell_id=n.cell_id, is_same_site=n.is_same_site, is_listed=n.is_listed,
                                       load_ratio=_clean(ratio), **fc,
                                       any_degraded=any(f.degraded for f in fc.values()),
                                       any_severe=any(f.severe for f in fc.values())))
    return PredictResponse(request_id=request_id, failed_cell=req.failed_cell.cell_id, prediction_time=slot,
                           model=p.bundle.get("deploy_model", "unknown"), app_version=settings.app_version,
                           degraded_threshold_pct=100 * settings.degraded_threshold,
                           severe_threshold_pct=100 * settings.severe_threshold,
                           n_neighbours=len(out), n_degraded=sum(o.any_degraded for o in out), neighbours=out)
