"""POST /predict: predictions from values supplied by the caller (any values, not only stored ones)."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..config import settings
from ..deps import get_predictor, request_id, require_api_key
from ..predictor import Predictor
from ..schemas import PredictRequest, PredictResponse
from .. import service

router = APIRouter(tags=["predict"], dependencies=[Depends(require_api_key)])
log = logging.getLogger("api")


def run(p: Predictor, req: PredictRequest, rid: str, actual: dict | None = None) -> PredictResponse:
    if len(req.neighbours) > settings.max_neighbours:
        raise HTTPException(status_code=422, detail=f"at most {settings.max_neighbours} neighbours per request")
    resp = service.predict(p, req, rid, actual)
    log.info("prediction", extra={"extra_fields": {"request_id": rid, "failed_cell": req.failed_cell.cell_id,
                                                   "neighbours": resp.n_neighbours, "degraded": resp.n_degraded}})
    return resp


@router.post("/predict", response_model=PredictResponse,
             summary="Predict every neighbour's latency, utilization and packet loss from supplied values")
def predict(req: PredictRequest, p: Predictor = Depends(get_predictor), rid: str = Depends(request_id)):
    return run(p, req, rid)


@router.get("/example", summary="A ready-made request body built from a notebook test row")
def example(index: int = 0):
    """Kept for the CI smoke test: works even without a data store."""
    try:
        cases = json.loads(settings.examples_path.read_text())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="no example file in this deployment")
    c = cases[index % len(cases)]["inputs"]
    lw = {k: v for k, v in (c.get("nb_last_week") or {}).items() if v is not None}
    others = max(c["n_absorbing"] - 1, 0)
    rest = (c["total_absorbing_capacity"] - c["nb_now"]["Capacity"]) / others if others else 0
    neighbours = [{"cell_id": "NBR-1", "is_same_site": bool(c["is_same_site"]), "is_listed": bool(c["is_listed"]),
                   "now": c["nb_now"], "h3_before": c["nb_3h"], "h6_before": c["nb_6h"], "last_week": lw or None}]
    for i in range(others):   # stand-ins carrying the rest of the absorbing capacity
        filler = dict(c["nb_now"], Capacity=rest)
        neighbours.append({"cell_id": f"NBR-{i + 2}", "is_same_site": False, "is_listed": True,
                           "now": filler, "h3_before": filler, "h6_before": filler, "last_week": None})
    return {"outage_start": f"2025-06-15T{c['hour']:02d}:00:00",
            "failed_cell": {"cell_id": "FAILED-1", **c["failed"], "site_cells_down": c["site_cells_down"]},
            "neighbours": neighbours}
