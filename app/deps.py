"""Shared state and FastAPI dependencies: loaded model, loaded data store, API-key check."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from .config import settings
from .predictor import Predictor
from .store import DataStore


@dataclass
class AppState:
    predictor: Predictor | None = None
    model_error: str | None = None
    data: DataStore | None = None
    data_error: str | None = None


state = AppState()


def require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="missing or invalid API key (send it in the X-API-Key header)")


def get_predictor() -> Predictor:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return state.predictor


def get_data() -> DataStore:
    if state.data is None:
        raise HTTPException(status_code=503, detail=f"data store not loaded from {settings.data_dir}: "
                                                    f"{state.data_error or 'unknown error'}")
    return state.data


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")
