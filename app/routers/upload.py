"""Endpoints for files the user uploads. Nothing is saved: each call reads the files it receives.

  file   (required)  KPI readings and OUTAGE/RECOVERY records in Dataset 2 format (CSV, or a ZIP with one CSV)
  cells  (optional)  NEW cells in Dataset 1 format: Site_ID, Cell_ID, Neighbor_Cell_ID (+ Technology, Region,
                     District). They are added to the stored Dataset 1 for this request only.
  add_reverse        also add each new cell to the lists of the cells it names (default true)
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..config import settings
from ..deps import get_data, get_predictor, request_id, require_api_key
from ..predictor import Predictor
from ..schemas import LookupResponse, PredictByCellResponse, PredictRequest
from ..store import DataStore
from ..store.dataset2 import UploadError, read_table, store_from_upload
from ..store.new_cells import add_new_cells, read_cells_table
from .lookup import do_lookup
from .predict import run

router = APIRouter(prefix="/upload", tags=["upload"], dependencies=[Depends(require_api_key)])


async def _read(f: UploadFile) -> bytes:
    content = await f.read()
    if len(content) > settings.max_upload_mb * 1e6:
        raise HTTPException(status_code=413, detail=f"{f.filename}: file larger than {settings.max_upload_mb:g} MB")
    return content


async def _prepare(file: UploadFile, cells: UploadFile | None, add_reverse: bool, ds: DataStore):
    """-> (KPI store from the file, topology incl. new cells, report)"""
    try:
        topo, cells_report = ds.topology, None
        if cells is not None and cells.filename:
            topo, cells_report = add_new_cells(ds.topology, read_cells_table(await _read(cells)), add_reverse)
        store, report = store_from_upload(read_table(await _read(file), file.filename or ""), ds.kpi.grid, topo)
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    report["new_cells"] = cells_report
    return store, topo, report


@router.post("/outages", summary="Read the files; list the outages and cells they contain")
async def upload_outages(file: UploadFile = File(...), cells: UploadFile | None = File(None),
                         add_reverse: bool = Form(True), ds: DataStore = Depends(get_data)):
    store, _, report = await _prepare(file, cells, add_reverse, ds)
    o = store.outages.sort_values("start")
    items = [{"cell_id": r.cell_id, "start": r.start, "up": None if r.up != r.up else r.up,
              "duration_h": None if r.duration_h != r.duration_h else r.duration_h} for r in o.itertuples()]
    return {"report": report, "outages": items, "cells": store.cells}


@router.post("/lookup", response_model=LookupResponse, summary="Build the request for one outage from the files")
async def upload_lookup(file: UploadFile = File(...), cell_id: str = Form(...), outage_start: datetime = Form(...),
                        cells: UploadFile | None = File(None), add_reverse: bool = Form(True),
                        ds: DataStore = Depends(get_data)):
    store, topo, _ = await _prepare(file, cells, add_reverse, ds)
    return do_lookup(store, topo, cell_id, outage_start, require_real=False)


@router.post("/predict-by-cell", response_model=PredictByCellResponse, summary="Look up and predict from the files")
async def upload_predict(file: UploadFile = File(...), cell_id: str = Form(...), outage_start: datetime = Form(...),
                         cells: UploadFile | None = File(None), add_reverse: bool = Form(True),
                         ds: DataStore = Depends(get_data), p: Predictor = Depends(get_predictor),
                         rid: str = Depends(request_id)):
    store, topo, _ = await _prepare(file, cells, add_reverse, ds)
    lk = do_lookup(store, topo, cell_id, outage_start, require_real=False)
    return PredictByCellResponse(lookup=lk, prediction=run(p, PredictRequest.model_validate(lk.request.model_dump()),
                                                           rid, lk.actual))
