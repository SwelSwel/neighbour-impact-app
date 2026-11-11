"""Thin client for the prediction API. Every page talks to the API only through these functions."""
from __future__ import annotations

from datetime import datetime

import requests
import streamlit as st

from .settings import API_KEY, API_URL, TIMEOUT_S

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


class ApiError(Exception):
    def __init__(self, status: int, detail: str, errors: list | None = None, request_id: str | None = None):
        super().__init__(detail)
        self.status, self.detail, self.errors, self.request_id = status, detail, errors or [], request_id


def call(method: str, path: str, **kw) -> dict | list:
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=TIMEOUT_S, **kw)
    except requests.RequestException as exc:
        raise ApiError(0, f"Cannot reach the prediction service at {API_URL}: {exc}")
    if r.status_code >= 400:
        try:
            body = r.json()
        except ValueError:
            body = {"detail": r.text[:300]}
        raise ApiError(r.status_code, str(body.get("detail")), body.get("errors"), body.get("request_id"))
    return r.json()


def show_error(exc: ApiError):
    if exc.errors:
        st.error(f"Some inputs are invalid ({exc.status}).")
        st.dataframe(exc.errors, hide_index=True)
    else:
        st.error(exc.detail if exc.status in (0, 404, 413, 422) else f"Request failed ({exc.status}): {exc.detail}")
    if exc.request_id:
        st.caption(f"Request ID {exc.request_id}")


# ---------------------------------------------------------------- endpoints
def ready() -> dict:
    try:
        return call("GET", "/ready")
    except ApiError as exc:
        return {"status": "not ready", "error": exc.detail}


@st.cache_data(ttl=120, show_spinner=False)
def model_info() -> dict:
    return call("GET", "/model-info")


@st.cache_data(ttl=120, show_spinner=False)
def data_info() -> dict:
    return call("GET", "/data-info")


@st.cache_data(ttl=300, show_spinner=False)
def outages(split: str = "", limit: int = 5000) -> list[dict]:
    return call("GET", "/outages", params={"split": split, "limit": limit})


@st.cache_data(ttl=300, show_spinner=False)
def search_cells(q: str, limit: int = 100) -> list[str]:
    return call("GET", "/cells", params={"q": q, "limit": limit})["cells"]


def lookup(cell_id: str, outage_start: datetime) -> dict:
    return call("POST", "/lookup", json={"cell_id": cell_id, "outage_start": outage_start.isoformat()})


def predict(request: dict) -> dict:
    return call("POST", "/predict", json=request)


def _upload_files(readings: tuple[str, bytes], new_cells: tuple[str, bytes] | None) -> dict:
    files = {"file": readings}
    if new_cells:
        files["cells"] = new_cells
    return files


def upload_outages(readings: tuple[str, bytes], new_cells: tuple[str, bytes] | None = None,
                   add_reverse: bool = True) -> dict:
    """readings / new_cells = (file name, file bytes). new_cells: Dataset 1-format file with new cells (optional)."""
    return call("POST", "/upload/outages", files=_upload_files(readings, new_cells),
                data={"add_reverse": str(add_reverse).lower()})


def upload_lookup(readings: tuple[str, bytes], cell_id: str, outage_start: datetime,
                  new_cells: tuple[str, bytes] | None = None, add_reverse: bool = True) -> dict:
    return call("POST", "/upload/lookup", files=_upload_files(readings, new_cells),
                data={"cell_id": cell_id, "outage_start": outage_start.isoformat(),
                      "add_reverse": str(add_reverse).lower()})
