"""Page 1: choose an outage (only the cell and the time), predict, compare with what happened, edit and re-predict."""
import pandas as pd
import streamlit as st

from lib import api, pickers
from lib.layout import case_section, new_case, page

KEY = "case_p1"
page("1 · Predict an outage",
     "Choose an outage. The app looks up every input from the data store, so only the cell and the time are needed.")

try:
    info = api.data_info()
except api.ApiError as exc:
    api.show_error(exc)
    st.stop()
if not info.get("loaded"):
    st.error(f"No data store is loaded ({info.get('error')}). Copy the notebook's app_data folder into the API's "
             f"data/ folder, or use page 3 (upload) or page 4 (API).")
    st.stop()

any_time = info["mode"] == "full" or info.get("allow_any_time")
how = st.radio("How do you want to choose the outage?",
               ["A real outage from the test period", "Any cell and time"] if any_time else
               ["A real outage from the test period"], horizontal=True)
if how.startswith("A real"):
    pick = pickers.real_outage(KEY)
else:
    pick = pickers.cell_and_time(KEY, pd.Timestamp(info["test_period_start"]).date())
    st.caption(f"Times from {info['period_start'][:16].replace('T', ' ')} to {info['period_end'][:16].replace('T', ' ')} "
               "can be looked up. Actual values are shown only where the data has a reading.")

if pick and st.button("Look up and predict", type="primary"):
    try:
        new_case(KEY, api.lookup(*pick))
    except api.ApiError as exc:
        api.show_error(exc)

st.divider()
case_section(KEY)
