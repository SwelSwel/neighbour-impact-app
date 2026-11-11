"""How predictions are shown: summary numbers, one table row per neighbour and KPI, warnings highlighted."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .settings import TARGETS, WARNING_COLOURS


def to_frame(pred: dict, actual: dict | None = None) -> pd.DataFrame:
    rows = []
    for n in pred["neighbours"]:
        act = (actual or {}).get(n["cell_id"]) or {}
        for t, unit in TARGETS.items():
            f = n[t]
            rows.append({"Neighbour": n["cell_id"],
                         "Type": ("same site" if n["is_same_site"] else "other site") +
                                 (", listed" if n["is_listed"] else ""),
                         "KPI": f"{t.replace('_', ' ')} ({unit})", "Before": f["before"], "Predicted": f["predicted"],
                         "80% range": f"{f['range_80_low']:.2f} to {f['range_80_high']:.2f}",
                         "Change %": f["change_pct"], "Actual": act.get(t),
                         "Warning": "Severe" if f["severe"] else ("Degraded" if f["degraded"] else ""),
                         "Load ratio": n["load_ratio"]})
    return pd.DataFrame(rows)


def show(pred: dict, actual: dict | None = None, key: str = "res"):
    df = to_frame(pred, actual)
    n, bad = pred["n_neighbours"], pred["n_degraded"]
    severe = sum(x["any_severe"] for x in pred["neighbours"])
    lat = df[df.KPI.str.startswith("Latency")]
    c = st.columns(4)
    c[0].metric("Predicted reading", pd.Timestamp(pred["prediction_time"]).strftime("%d %b, %H:%M"))
    c[1].metric("Neighbours", n)
    c[2].metric(f"Degraded (> +{pred['degraded_threshold_pct']:.0f}%)", bad)
    c[3].metric(f"Severe (> +{pred['severe_threshold_pct']:.0f}%)", severe)

    has_actual = actual is not None and df["Actual"].notna().any()
    if has_actual:
        err = (df.Predicted - df.Actual).abs().groupby(df.KPI).mean()
        st.caption("Actual values recorded at that reading are shown for comparison. Mean absolute error on this "
                   "outage: " + ", ".join(f"{k} {v:.3g}" for k, v in err.items()) + ".")
    only = st.toggle("Only neighbours with a warning", key=f"{key}_only")
    if only:
        df = df[df.Neighbour.isin(df.loc[df.Warning != "", "Neighbour"])]
    cols = ["Neighbour", "Type", "KPI", "Before", "Predicted", "80% range", "Change %"] + \
           (["Actual"] if has_actual else []) + ["Warning"]

    def colour(r):
        c = WARNING_COLOURS.get(r["Warning"].lower(), "")
        return [f"background-color: {c}" if c else "" for _ in r]
    st.dataframe(df[cols].style.apply(colour, axis=1).format(
        {"Before": "{:.2f}", "Predicted": "{:.2f}", "Change %": "{:+.1f}", "Actual": "{:.2f}"}, na_rep="–"),
        width="stretch", hide_index=True, height=min(38 * (len(df) + 1), 560))
    if not lat.empty:
        worst = lat.sort_values("Change %", ascending=False).iloc[0]
        if pd.notna(worst["Change %"]):
            st.caption(f"Largest predicted latency increase: {worst.Neighbour}, {worst['Change %']:+.1f}% "
                       f"({worst.Before:.1f} → {worst.Predicted:.1f} ms). Model {pred['model']}, "
                       f"version {pred['app_version']}, request {pred['request_id']}.")
    st.download_button("Download results (CSV)", to_frame(pred, actual).to_csv(index=False),
                       file_name=f"prediction_{pred['failed_cell']}.csv", key=f"{key}_dl")


def mean_change(pred: dict) -> dict[str, float]:
    """Average predicted % change per KPI over all neighbours (used by the what-if page)."""
    out = {}
    for t in TARGETS:
        vals = [n[t]["change_pct"] for n in pred["neighbours"] if n[t]["change_pct"] is not None]
        out[t] = float(np.mean(vals)) if vals else np.nan
    return out
