"""Read a file in Dataset 2 format (the monthly CSVs) into a KPIStore, with the notebook's parsing rules."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from .grid import Grid
from .kpi_store import RAW8, KPIStore
from .outages import down_slots, pair_events
from .topology import Topology

REQUIRED = ["Timestamp", "Cell_ID", "Record_Type"] + RAW8
OPTIONAL = ["Site_ID", "Status", "Cell_Down_Time", "Cell_Up_Time"]


class UploadError(ValueError):
    pass


def parse_ts(s: pd.Series) -> pd.Series:
    """Notebook's parse_ts: a bare date (10 characters) means midnight."""
    s = s.astype("string").str.strip()
    s = s.where(s.str.len() != 10, s + " 00:00:00")
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def read_table(content: bytes, filename: str = "") -> pd.DataFrame:
    """CSV, or a ZIP containing one CSV."""
    if filename.lower().endswith(".zip") or content[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise UploadError("the ZIP file contains no .csv file")
                content = z.read(names[0])
        except zipfile.BadZipFile as exc:
            raise UploadError(f"could not read the file as ZIP: {exc}") from exc
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, usecols=lambda c: c in REQUIRED + OPTIONAL)
    except Exception as exc:
        raise UploadError(f"could not read the file as CSV: {exc}") from exc
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise UploadError(f"missing columns: {', '.join(missing)} (expected the Dataset 2 format)")
    return df


def store_from_upload(df: pd.DataFrame, grid: Grid, topology: Topology) -> tuple[KPIStore, dict]:
    """Build a KPIStore from an uploaded Dataset 2 table. Returns the store and a short parsing report."""
    rt = df["Record_Type"].astype(str).str.upper().str.strip()
    ts = parse_ts(df["Timestamp"])
    cid = df["Cell_ID"].astype(str).str.strip()
    report = {"rows": int(len(df)), "kpi_rows": int(rt.eq("KPI").sum()),
              "outage_records": int(rt.eq("OUTAGE").sum()), "recovery_records": int(rt.eq("RECOVERY").sum()),
              "unparseable_timestamps": int(ts.isna().sum())}

    # KPI readings on the 3-hourly grid, for cells that exist in Dataset 1
    k = rt.eq("KPI") & ts.notna() & cid.map(topology.has)
    report["unknown_cells"] = int((rt.eq("KPI") & ~cid.map(topology.has)).sum())
    kp = df[k].copy()
    kp["cell_id"] = cid[k]
    off = (ts[k] - pd.Timestamp(grid.start)) / grid.step
    on_grid = off == np.floor(off)
    report["off_grid_rows"] = int((~on_grid).sum())
    kp = kp[on_grid.to_numpy()]
    kp["slot"] = off[on_grid].astype(np.int64).to_numpy()
    for v in RAW8:
        kp[v] = pd.to_numeric(kp[v], errors="coerce")
    kp = kp.drop_duplicates(["cell_id", "slot"], keep="last")[["cell_id", "slot"] + RAW8]
    if kp.empty:
        raise UploadError("no usable KPI rows: check Record_Type = KPI, the timestamps and the Cell_ID values")

    # Outages from OUTAGE / RECOVERY records, and the "down" flag
    ev = df[rt.isin(["OUTAGE", "RECOVERY"])].copy()
    if len(ev):
        ev["Record_Type"] = rt[ev.index]
        ev["Timestamp"] = ts[ev.index]
        for c in ["Cell_Down_Time", "Cell_Up_Time"]:
            ev[c] = parse_ts(ev[c]) if c in ev else pd.NaT
    outages = pair_events(ev) if len(ev) else pair_events(pd.DataFrame())
    slots_by_cell = {c: np.sort(g.slot.to_numpy()) for c, g in kp.groupby("cell_id")}
    dn = down_slots(outages, grid, slots_by_cell, int(kp.slot.max()))
    kp["down"] = [(c, s) in dn for c, s in zip(kp.cell_id, kp.slot)]

    outages = outages.assign(event_id=None, split="upload", in_test_period=True, n_model_rows=0)
    store = KPIStore(kp.reset_index(drop=True), grid, outages.reset_index(drop=True),
                     {"mode": "upload", "week_slots": 7 * 24 // grid.step_h}, "upload")
    report.update({"cells": int(kp.cell_id.nunique()), "outages": int(len(outages)),
                   "period_start": str(store.period[0]), "period_end": str(store.period[1])})
    return store, report
