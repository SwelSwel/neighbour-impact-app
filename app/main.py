"""FastAPI application: start-up, logging, error handling and the routers.

Endpoints (see /docs for the interactive version)
  system      GET  /health  /ready  /model-info  /data-info
  predict     POST /predict            any values, supplied by the caller
              GET  /example            a ready-made request body
  data store  GET  /cells  /cells/{id}  /outages
              POST /lookup             cell + time -> the full request, to edit and send to /predict
              POST /predict-by-cell    cell + time -> predictions (+ actual values when known)
  upload      POST /upload/outages  /upload/lookup  /upload/predict-by-cell   (Dataset 2 file)
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .deps import state
from .predictor import Predictor
from .routers import lookup, predict, system, upload
from .store import DataStore


# ----------------------------------------------------------------------------- logging: one JSON object per line
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                 "logger": record.name, "message": record.getMessage()}
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
log = logging.getLogger("api")


# ----------------------------------------------------------------------------- start-up: load model and data once
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.predictor = Predictor(settings.model_path)
    except Exception as exc:   # keep running so /health and /ready can report the problem
        state.model_error = repr(exc)
        log.exception("model bundle failed to load from %s", settings.model_path)
    try:
        t = time.perf_counter()
        state.data = DataStore.from_dir(settings.data_dir)
        log.info("data store loaded", extra={"extra_fields": {**state.data.info(),
                                                              "seconds": round(time.perf_counter() - t, 1)}})
    except Exception as exc:   # /predict still works without it
        state.data_error = repr(exc)
        log.warning("data store not loaded from %s: %s", settings.data_dir, exc)
    yield


app = FastAPI(title="Neighbour Impact Prediction API", version=settings.app_version, lifespan=lifespan,
              description="Predicts each neighbouring cell's latency, utilization and packet loss when a cell goes "
                          "down. Send the API key in the `X-API-Key` header.")


# ----------------------------------------------------------------------------- middleware and error handling
@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = rid
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"extra_fields": {"request_id": rid, "path": request.url.path}})
        response = JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    log.info("request", extra={"extra_fields": {
        "request_id": rid, "method": request.method, "path": request.url.path,
        "status": response.status_code, "duration_ms": round(1000 * (time.perf_counter() - start), 1)}})
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    log.warning("invalid request", extra={"extra_fields": {"request_id": request.state.request_id, "errors": errors[:5]}})
    return JSONResponse(status_code=422, content={"detail": "invalid input", "errors": errors,
                                                  "request_id": request.state.request_id})


for r in (system.router, predict.router, lookup.router, upload.router):
    app.include_router(r)
