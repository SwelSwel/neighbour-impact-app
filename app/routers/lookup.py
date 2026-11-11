"""Endpoints that read the data store: cell search, outage list, lookup and predict-by-cell."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..deps import get_data, get_predictor, request_id, require_api_key
from ..predictor import Predictor
from ..schemas import CellTimeRequest, LookupResponse, OutageItem, PredictByCellResponse, PredictRequest
from ..store import DataStore
from ..store.lookup import LookupFailed, lookup
from .predict import run

router = APIRouter(tags=["data store"], dependencies=[Depends(require_api_key)])


def do_lookup(ds_kpi, topo, cell_id: str, start: datetime, require_real: bool) -> LookupResponse:
    try:
        return LookupResponse(**lookup(ds_kpi, topo, cell_id, start, require_real_outage=require_real))
    except LookupFailed as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message)


def _require_real(ds: DataStore) -> bool:
    return ds.mode == "outages" and not settings.allow_any_time


@router.get("/cells", summary="Search cell IDs")
def cells(q: str = "", limit: int = Query(50, ge=1, le=500), ds: DataStore = Depends(get_data)):
    return {"cells": ds.topology.search(q, limit)}


@router.get("/cells/{cell_id}", summary="One cell: site, technology, region and its neighbours")
def cell(cell_id: str, ds: DataStore = Depends(get_data)):
    if not ds.topology.has(cell_id):
        raise HTTPException(status_code=404, detail=f"cell {cell_id} not found")
    return {**ds.topology.info(cell_id),
            "neighbours": [vars(n) for n in ds.topology.neighbours(cell_id)]}


@router.get("/outages", response_model=list[OutageItem], summary="Real outages in the data store (test period)")
def outages(split: str = Query("", description="'test', 'unseen site', or empty for all"),
            q: str = Query("", description="part of a cell ID"), technology: str = "", region: str = "",
            min_neighbours: int = 0, limit: int = Query(500, ge=1, le=5000), ds: DataStore = Depends(get_data)):
    o = ds.kpi.outages
    if "in_test_period" in o:
        o = o[o.in_test_period.astype(bool)]
    if "split" in o:
        o = o[o.split != ""] if not split else o[o.split == split]
    if q:
        o = o[o.cell_id.str.contains(q, case=False, regex=False)]
    cells = ds.topology.cells
    o = o.assign(technology=o.cell_id.map(cells["technology"]), region=o.cell_id.map(cells["region"]))
    if technology:
        o = o[o.technology == technology]
    if region:
        o = o[o.region == region]
    if min_neighbours and "n_model_rows" in o:
        o = o[o.n_model_rows >= min_neighbours]
    o = o.sort_values("start").head(limit)
    na = lambda v: None if pd.isna(v) else v
    return [OutageItem(event_id=na(r.get("event_id")), cell_id=r["cell_id"], start=r["start"], up=na(r["up"]),
                       duration_h=na(r["duration_h"]), split=r.get("split", ""), technology=na(r["technology"]),
                       region=na(r["region"]), n_neighbours=na(r.get("n_model_rows")))
            for r in o.to_dict("records")]


@router.post("/lookup", response_model=LookupResponse,
             summary="Build the full request for an outage from the data store (edit it, then POST /predict)")
def lookup_endpoint(body: CellTimeRequest, ds: DataStore = Depends(get_data)):
    return do_lookup(ds.kpi, ds.topology, body.cell_id, body.outage_start, _require_real(ds))


@router.post("/predict-by-cell", response_model=PredictByCellResponse,
             summary="Only a cell and a time: look up every input, predict, and show actual values when known")
def predict_by_cell(body: CellTimeRequest, ds: DataStore = Depends(get_data),
                    p: Predictor = Depends(get_predictor), rid: str = Depends(request_id)):
    lk = do_lookup(ds.kpi, ds.topology, body.cell_id, body.outage_start, _require_real(ds))
    pred = run(p, PredictRequest.model_validate(lk.request.model_dump()), rid, lk.actual)
    return PredictByCellResponse(lookup=lk, prediction=pred)
