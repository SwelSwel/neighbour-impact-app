"""Page 3: predict from the user's own files.

Option 1: readings only (Dataset 2 format), for cells that are already in our network (Dataset 1).
Option 2: readings + new cells (Dataset 1 format), for cells that are not in our network yet.
Both files can be downloaded as templates with example rows.
"""
from datetime import datetime, time

import pandas as pd
import streamlit as st

from lib import api, templates
from lib.layout import case_section, new_case, page

KEY = "case_p3"
page("3 · Upload data",
     "Predict from your own KPI readings. The cells can be ones already in our network, or new cells you describe.")

MODE_EXISTING = "Cells already in our network"
MODE_NEW = "Includes new cells"
mode = st.radio("What are you uploading?", [MODE_EXISTING, MODE_NEW], horizontal=True,
                captions=["One file: KPI readings",
                          "Two files: KPI readings + the new cells' details"])


# ---------------------------------------------------------------- templates and file formats
@st.cache_data(ttl=3600, show_spinner=False)
def example_cells() -> tuple[str, str, list[str]]:
    """A real cell, one of its neighbours, and a few existing cell IDs, so the templates use valid IDs."""
    try:
        cid = api.search_cells("", 1)[0]
        nbrs = [n["cell_id"] for n in api.call("GET", f"/cells/{cid}")["neighbours"]]
        return cid, nbrs[0], [cid] + nbrs[:2]
    except (api.ApiError, IndexError):
        return "<failed cell ID>", "<neighbour cell ID>", []


failed_ex, nbr_ex, existing_ex = example_cells()
with st.expander("File formats and templates", expanded=False):
    st.markdown("**KPI readings**")
    st.dataframe(templates.READING_HELP, hide_index=True, width="stretch")
    st.markdown("Include the failed cell **and its neighbours**, from at least 6 hours before the outage. Add the "
                "week before it to get the last-week features; without them they count as missing, as in training.")
    if mode == MODE_EXISTING:
        st.download_button("Download readings template (CSV)", templates.readings_template(failed_ex, nbr_ex),
                           file_name="readings_template.csv", mime="text/csv")
    else:
        st.download_button("Download readings template (CSV)", templates.readings_template("UG-NEW-0001-S01",
                           existing_ex[0] if existing_ex else nbr_ex), file_name="readings_template.csv",
                           mime="text/csv")
        st.markdown("**New cells** (cell information file)")
        st.dataframe(templates.CELL_HELP, hide_index=True, width="stretch")
        st.download_button("Download new-cells template (CSV)", templates.new_cells_template(existing_ex),
                           file_name="new_cells_template.csv", mime="text/csv")
    st.caption("The templates contain example rows: replace them with your own values. Cell IDs in the examples are "
               "real cells from our network, so the neighbour links work.")

# ---------------------------------------------------------------- uploads
c1, c2 = st.columns(2)
readings = c1.file_uploader("KPI readings (CSV or ZIP)", type=["csv", "zip"], key="up_readings")
new_cells, add_reverse = None, True
if mode == MODE_NEW:
    new_cells = c2.file_uploader("New cells (CSV)", type=["csv"], key="up_cells")
    add_reverse = c2.checkbox("Also add the new cells to their neighbours' lists", value=True,
                              help="Then a new cell can take users when one of the existing cells it lists goes "
                                   "down. Untick to use only the new cells' own lists.")
if not readings or (mode == MODE_NEW and not new_cells):
    st.info("Upload " + ("the readings file." if mode == MODE_EXISTING else "both files."))
    st.stop()

r_file = (readings.name, readings.getvalue())
c_file = (new_cells.name, new_cells.getvalue()) if new_cells else None
sig = (mode, r_file[0], len(r_file[1]), c_file and (c_file[0], len(c_file[1])), add_reverse)
if st.session_state.get("upload_sig") != sig:
    try:
        st.session_state["upload_info"] = api.upload_outages(r_file, c_file, add_reverse)
        st.session_state["upload_sig"] = sig
        st.session_state.pop(KEY, None)
    except api.ApiError as exc:
        api.show_error(exc)
        st.stop()
info = st.session_state["upload_info"]
rep = info["report"]

st.success(f"Read {rep['rows']:,} rows: {rep['kpi_rows']:,} KPI readings for {rep['cells']} cells, "
           f"{rep['outages']} outages, from {rep['period_start'][:16]} to {rep['period_end'][:16]}.")
if rep["unknown_cells"] or rep["off_grid_rows"] or rep["unparseable_timestamps"]:
    st.warning(f"Ignored: {rep['unknown_cells']} readings of cells that are not in Dataset 1"
               + (" or the new-cells file" if mode == MODE_NEW else "") +
               f", {rep['off_grid_rows']} readings off the 3-hourly grid, "
               f"{rep['unparseable_timestamps']} unreadable timestamps.")
nc = rep.get("new_cells")
if nc:
    n_new = len(nc["new_cells"])
    st.success(f"Added {n_new} new cell{'s' if n_new != 1 else ''}"
               + (f" on {len(nc['new_sites'])} new sites" if nc["new_sites"] else "")
               + f": {nc['listed_links']} listed neighbour links"
               + (f", plus {nc['reverse_links']} links back from those neighbours" if nc["reverse_links"] else "")
               + ".")
    st.dataframe(pd.DataFrame({"New cell": list(nc["neighbours_per_new_cell"]),
                               "Neighbours": list(nc["neighbours_per_new_cell"].values())}), hide_index=True)
    if nc["n_unknown_neighbour_ids"]:
        st.warning(f"{nc['n_unknown_neighbour_ids']} neighbour IDs are in neither Dataset 1 nor the new-cells file "
                   f"and were ignored: {', '.join(nc['unknown_neighbour_ids'][:10])}")
new_ids = set(nc["new_cells"]) if nc else set()
tag = lambda c: f"{c}  (new)" if c in new_ids else c

# ---------------------------------------------------------------- choose the outage
options = ["An outage recorded in the file", "A cell and a time"] if info["outages"] else ["A cell and a time"]
how = st.radio("Which outage?", options, horizontal=True)
if how.startswith("An outage"):
    o = pd.DataFrame(info["outages"])
    o["start"] = pd.to_datetime(o.start)
    pick = st.selectbox("Outage", list(o.itertuples(index=False)),
                        format_func=lambda r: f"{r.start:%Y-%m-%d %H:%M} · {tag(r.cell_id)}"
                                              + (f" · {r.duration_h:.1f} h" if pd.notna(r.duration_h) else ""))
    cell, start = pick.cell_id, pick.start.to_pydatetime()
else:
    c = st.columns(3)
    cell = c[0].selectbox("Failed cell", info["cells"], format_func=tag)
    day = c[1].date_input("Outage date", value=pd.Timestamp(rep["period_end"]).date())
    start = datetime.combine(day, c[2].time_input("Start time", value=time(12, 0), step=600))

if st.button("Look up and predict", type="primary"):
    try:
        new_case(KEY, api.upload_lookup(r_file, cell, start, c_file, add_reverse))
    except api.ApiError as exc:
        api.show_error(exc)

st.divider()
case_section(KEY)
