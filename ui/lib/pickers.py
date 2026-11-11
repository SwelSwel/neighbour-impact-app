"""Ways to choose an outage: from the list of real ones, or any cell and time."""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import streamlit as st

from . import api


def real_outage(key: str) -> tuple[str, datetime] | None:
    """Filterable list of real outages from the data store. Returns (cell_id, start) or None."""
    try:
        rows = api.outages()
    except api.ApiError as exc:
        api.show_error(exc)
        return None
    if not rows:
        st.info("The data store has no listed outages.")
        return None
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df.start)
    c = st.columns(4)
    split = c[0].selectbox("Set", ["all"] + sorted(df.split.dropna().unique()), key=f"{key}_split",
                           help="test = development sites, June. unseen site = sites never used in training.")
    tech = c[1].selectbox("Technology", ["all"] + sorted(df.technology.dropna().unique()), key=f"{key}_tech")
    region = c[2].selectbox("Region", ["all"] + sorted(df.region.dropna().unique()), key=f"{key}_reg")
    q = c[3].text_input("Cell ID contains", key=f"{key}_q")
    if split != "all":
        df = df[df.split == split]
    if tech != "all":
        df = df[df.technology == tech]
    if region != "all":
        df = df[df.region == region]
    if q:
        df = df[df.cell_id.str.contains(q, case=False, regex=False)]
    if df.empty:
        st.warning("No outage matches these filters.")
        return None
    label = lambda r: (f"{r.start:%Y-%m-%d %H:%M} · {r.cell_id} · {r.technology or '?'} · {r.region or '?'} · "
                       f"{r.n_neighbours or '?'} neighbours · {r.split}")
    options = list(df.itertuples(index=False))
    pick = st.selectbox(f"Outage ({len(options):,} match)", options, format_func=label, key=f"{key}_pick")
    return pick.cell_id, pick.start.to_pydatetime()


def cell_and_time(key: str, default_day=None) -> tuple[str, datetime] | None:
    """Any cell and any time (needs a 'full' data store, or ALLOW_ANY_TIME)."""
    c = st.columns([2, 2, 1, 1])
    q = c[0].text_input("Search cell ID", key=f"{key}_cq")
    try:
        cells = api.search_cells(q)
    except api.ApiError as exc:
        api.show_error(exc)
        return None
    if not cells:
        st.warning("No cell matches.")
        return None
    cell = c[1].selectbox("Cell", cells, key=f"{key}_cell")
    day = c[2].date_input("Outage date", value=default_day, key=f"{key}_day")
    t = c[3].time_input("Start time", value=time(12, 0), step=600, key=f"{key}_time")
    return (cell, datetime.combine(day, t)) if day else None
