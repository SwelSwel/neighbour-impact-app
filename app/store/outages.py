"""Outage events: pairing OUTAGE with RECOVERY records, the "down" rule and site-level counts.

Same rules as notebook section 5, so an uploaded file is treated exactly like the training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import Grid


def pair_events(events: pd.DataFrame) -> pd.DataFrame:
    """events: Cell_ID, Record_Type (OUTAGE/RECOVERY), Timestamp, Cell_Down_Time, Cell_Up_Time (datetimes).

    Returns one row per outage: cell_id, start, up (NaT if no recovery), duration_h."""
    cols = ["cell_id", "start", "up", "duration_h"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    ev = events.copy()
    ev["Cell_ID"] = ev["Cell_ID"].astype(str).str.strip()
    out = ev[ev.Record_Type.eq("OUTAGE")].copy()
    out["start"] = out["Cell_Down_Time"].fillna(out["Timestamp"])
    out = out.dropna(subset=["start"]).drop_duplicates(["Cell_ID", "start"]).sort_values("start")
    rec = ev[ev.Record_Type.eq("RECOVERY")].copy()
    rec["up"] = rec["Cell_Up_Time"].fillna(rec["Timestamp"])
    rec = rec.dropna(subset=["up"]).drop_duplicates(["Cell_ID", "up"]).sort_values("up")
    if out.empty:
        return pd.DataFrame(columns=cols)
    if rec.empty:
        out["up"] = pd.NaT
    else:
        out = pd.merge_asof(out, rec[["Cell_ID", "up"]], left_on="start", right_on="up", by="Cell_ID",
                            direction="forward")
    out = out.sort_values(["Cell_ID", "start"]).reset_index(drop=True)
    nxt = out.groupby("Cell_ID")["start"].shift(-1)
    out.loc[nxt.notna() & out.up.notna() & (out.up > nxt), "up"] = pd.NaT   # recovery belongs to a later outage
    out["duration_h"] = (out.up - out.start).dt.total_seconds() / 3600
    return out.rename(columns={"Cell_ID": "cell_id"})[cols]


def down_slots(outages: pd.DataFrame, grid: Grid, reading_slots: dict[str, np.ndarray], last_slot: int) -> set[tuple[str, int]]:
    """(cell, slot) pairs during which the cell counts as down.

    From ceil(start) up to ceil(recovery); with no recovery, until the cell's next KPI reading (notebook rule)."""
    down: set[tuple[str, int]] = set()
    for cid, start, up in zip(outages.cell_id, outages.start, outages.up):
        a = grid.slot_ceil(start)
        if pd.isna(up):
            later = reading_slots.get(cid, np.array([], int))
            later = later[later >= a]
            u = int(later[0]) if len(later) else last_slot + 1
        else:
            u = grid.slot_ceil(up)
        down.update((cid, s) for s in range(a, max(u, a)))
    return down


def site_cells_down(outages: pd.DataFrame, cells_on_site: set[str], cell_id: str, t) -> int:
    """Other cells on the failed cell's site whose outage started at or before t and had not recovered by t."""
    if outages.empty:
        return 0
    o = outages[outages.cell_id.isin(cells_on_site - {cell_id})]
    t = pd.Timestamp(t)
    return int(((o.start <= t) & (o.up.isna() | (o.up > t))).sum())
