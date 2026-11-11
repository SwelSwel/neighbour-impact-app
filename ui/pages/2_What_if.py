"""Page 2: start from a looked-up outage and change a few drivers of the impact with sliders."""
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from lib import api, pickers, results
from lib.layout import new_case, page
from lib.settings import TARGETS

KEY = "case_p2"
page("2 · What-if",
     "Start from a real outage, then change the failed cell's load, the cells already down on its site or the time "
     "of day. Every change is a new set of inputs, so the model predicts values it has never seen.")

# ---------------------------------------------------------------- base case: page 1's outage, or pick one here
base = st.session_state.get("case_p1") if st.toggle("Use the outage from page 1", value="case_p1" in st.session_state,
                                                   disabled="case_p1" not in st.session_state) else None
if base is None:
    with st.expander("Choose an outage", expanded=KEY not in st.session_state):
        pick = pickers.real_outage(KEY)
        if pick and st.button("Use this outage"):
            try:
                new_case(KEY, api.lookup(*pick), auto_predict=False)
            except api.ApiError as exc:
                api.show_error(exc)
    base = st.session_state.get(KEY)
if not base:
    st.stop()
req0 = base["lookup"]["request"]
fc0 = req0["failed_cell"]
start0 = datetime.fromisoformat(str(req0["outage_start"]))
st.write(f"Outage of **{fc0['cell_id']}** at {start0:%Y-%m-%d %H:%M}, {len(req0['neighbours'])} neighbours, "
         f"failed cell load {fc0.get('Load') or 0:.1f} (capacity {fc0.get('Capacity') or 0:.1f}).")


# ---------------------------------------------------------------- scenario controls
def scenario(req: dict, load_x: float, site_down: int, hour: int, drop: list[str]) -> dict:
    r = copy.deepcopy(req)
    if r["failed_cell"].get("Load") is not None:
        r["failed_cell"]["Load"] = r["failed_cell"]["Load"] * load_x
    r["failed_cell"]["site_cells_down"] = site_down
    r["outage_start"] = start0.replace(hour=hour, minute=0).isoformat() if hour != start0.hour else start0.isoformat()
    r["neighbours"] = [n for n in r["neighbours"] if n["cell_id"] not in drop] or r["neighbours"][:1]
    return r


c = st.columns(4)
load_x = c[0].slider("Failed cell's load (× recorded)", 0.0, 3.0, 1.0, 0.1)
site_down = c[1].number_input("Other cells on the site already down", 0, 20, int(fc0.get("site_cells_down") or 0))
hour = c[2].select_slider("Outage hour", list(range(24)), value=start0.hour)
drop = c[3].multiselect("Neighbours also down", [n["cell_id"] for n in req0["neighbours"]],
                        help="Removed neighbours no longer share the load.")

try:
    base_pred = api.predict(req0)
    scen_pred = api.predict(scenario(req0, load_x, site_down, hour, drop))
except api.ApiError as exc:
    api.show_error(exc)
    st.stop()

b, s = results.mean_change(base_pred), results.mean_change(scen_pred)
m = st.columns(4)
for col, t in zip(m[:3], TARGETS):
    col.metric(f"Mean predicted change: {t.replace('_', ' ')}", f"{s[t]:+.1f}%", f"{s[t] - b[t]:+.1f} pts vs recorded")
m[3].metric("Degraded neighbours", scen_pred["n_degraded"], scen_pred["n_degraded"] - base_pred["n_degraded"])

st.subheader("Scenario predictions")
results.show(scen_pred, key="whatif")

# ---------------------------------------------------------------- sweep over the failed cell's load
st.subheader("How the impact grows with the failed cell's load")
if st.button("Run the load sweep (0× to 3×)"):
    rows = []
    for x in np.round(np.arange(0, 3.01, 0.25), 2):
        pr = api.predict(scenario(req0, float(x), site_down, hour, drop))
        rows.append({"Load ×": x, **{t.replace("_", " "): v for t, v in results.mean_change(pr).items()},
                     "Degraded neighbours": pr["n_degraded"]})
    st.session_state["sweep"] = pd.DataFrame(rows).set_index("Load ×")
if "sweep" in st.session_state:
    sw = st.session_state["sweep"]
    c1, c2 = st.columns(2)
    c1.line_chart(sw[[t.replace("_", " ") for t in TARGETS]], x_label="Failed cell's load (× recorded)",
                  y_label="Mean predicted change (%)")
    c2.bar_chart(sw["Degraded neighbours"], x_label="Failed cell's load (× recorded)", y_label="Degraded neighbours")
    st.caption("Each point is a new prediction with only the failed cell's load changed. This is the dose-response "
               "pattern from the analysis: more displaced load, more impact on the neighbours.")
