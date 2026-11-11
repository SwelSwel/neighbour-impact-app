"""Health, readiness and information endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..config import settings
from ..deps import get_predictor, require_api_key, state
from ..predictor import Predictor

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness: the process is up")
def health():
    return {"status": "ok", "version": settings.app_version}


@router.get("/ready", summary="Readiness: the model is loaded (the data store is reported, not required)")
def ready():
    body = {"model": "loaded" if state.predictor else "not loaded",
            "data_store": "loaded" if state.data else "not loaded"}
    if state.predictor is None:
        return JSONResponse(status_code=503, content={"status": "not ready", "error": state.model_error, **body})
    return {"status": "ready", **body}


@router.get("/model-info", dependencies=[Depends(require_api_key)], summary="Model, versions and test metrics")
def model_info(p: Predictor = Depends(get_predictor)):
    return {**p.info(), "app_version": settings.app_version,
            "degraded_threshold": settings.degraded_threshold, "severe_threshold": settings.severe_threshold}


@router.get("/data-info", dependencies=[Depends(require_api_key)], summary="What the data store holds")
def data_info():
    if state.data is None:
        return {"loaded": False, "error": state.data_error, "data_dir": str(settings.data_dir)}
    return {"loaded": True, **state.data.info(), "allow_any_time": settings.allow_any_time}
