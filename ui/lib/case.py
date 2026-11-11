"""The editable "case": everything the model needs for one outage, shown as tables the user can change.

A case is the JSON body of POST /predict. A lookup fills it from the data; the user may change any value, add or
remove neighbours, and predict again. When nothing was changed, the actual values from the data are shown too.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from .settings import FAILED_KPIS, LAST_WEEK, RAW8, READINGS


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def fingerprint(request: dict) -> str:
    """Stable text form of a request, to tell whether the user changed anything.

    Missing readings and blank values compare equal; numbers are compared to 4 significant digits."""
    def reading(d, keys):
        return {k: (None if d is None or d.get(k) is None else float(f"{d[k]:.4g}")) for k in keys}
    norm = {"outage_start": str(request["outage_start"])[:16].replace("T", " "),
            "failed_cell": {**reading(request["failed_cell"], FAILED_KPIS),
                            "cell_id": request["failed_cell"]["cell_id"],
                            "site_cells_down": int(request["failed_cell"].get("site_cells_down") or 0)},
            "neighbours": [{"cell_id": n["cell_id"], "same": bool(n["is_same_site"]), "listed": bool(n["is_listed"]),
                            **{rk: reading(n.get(rk), RAW8) for rk, _ in READINGS},
                            "last_week": reading(n.get("last_week"), LAST_WEEK)} for n in request["neighbours"]]}
    return json.dumps(norm, default=str)


def editor(request: dict, key: str) -> dict:
    """Show the case as editable inputs and return the (possibly edited) request."""
    fc = dict(request["failed_cell"])
    nbs = request["neighbours"]
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        st.markdown("**Failed cell** (last reading before the outage)")
        start = st.text_input("Outage start (YYYY-MM-DD HH:MM)", str(request["outage_start"]).replace("T", " ")[:16],
                              key=f"{key}_start")
        st.text_input("Cell ID", fc["cell_id"], key=f"{key}_fcid", disabled=True)
        for k in FAILED_KPIS:
            fc[k] = st.number_input(k, value=_num(fc.get(k)), min_value=0.0, format="%.3f", key=f"{key}_fc_{k}")
        fc["site_cells_down"] = int(st.number_input("Other cells on this site already down", 0, 50,
                                                    int(fc.get("site_cells_down") or 0), key=f"{key}_scd"))
    with c2:
        st.markdown(f"**Neighbours that were up just before the outage** ({len(nbs)}). "
                    "Add or delete rows to change who shares the load.")
        meta = pd.DataFrame([{"cell_id": n["cell_id"], "is_same_site": n["is_same_site"], "is_listed": n["is_listed"]}
                             for n in nbs])
        meta = st.data_editor(meta, num_rows="dynamic", width="stretch", key=f"{key}_meta",
                              column_config={"is_same_site": st.column_config.CheckboxColumn("Same site"),
                                             "is_listed": st.column_config.CheckboxColumn("Listed neighbour")})
        tabs = st.tabs([label for _, label in READINGS] + ["Same time last week"])
        tables = {}
        for tab, (rk, _) in zip(tabs, READINGS):
            with tab:
                df = pd.DataFrame([n.get(rk) or {} for n in nbs], columns=RAW8)
                tables[rk] = st.data_editor(df.reindex(range(len(meta))), width="stretch", key=f"{key}_t_{rk}")
        with tabs[-1]:
            lw = pd.DataFrame([n.get("last_week") or {} for n in nbs], columns=LAST_WEEK)
            tables["last_week"] = st.data_editor(lw.reindex(range(len(meta))), width="stretch", key=f"{key}_t_lw")
            st.caption("Blank cells are allowed: the model treats them as missing, as in training.")

    try:
        original = str(request["outage_start"]).replace("T", " ")
        outage_start = (str(request["outage_start"]) if start.strip() == original[:16] else
                        datetime.strptime(start.strip(), "%Y-%m-%d %H:%M").isoformat())
    except ValueError:
        st.error("Outage start must look like 2025-06-15 10:40")
        outage_start = request["outage_start"]
    neighbours = []
    for i, m in meta.reset_index(drop=True).iterrows():
        row = lambda rk: {k: _num(v) for k, v in tables[rk].iloc[i].items()}
        lw_row = {k: v for k, v in row("last_week").items() if v is not None}
        neighbours.append({"cell_id": str(m["cell_id"]), "is_same_site": bool(m["is_same_site"]),
                           "is_listed": bool(m["is_listed"]), "now": row("now"), "h3_before": row("h3_before"),
                           "h6_before": row("h6_before"), "last_week": lw_row or None})
    return {"outage_start": outage_start, "failed_cell": fc, "neighbours": neighbours}
