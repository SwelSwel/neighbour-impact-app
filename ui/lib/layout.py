"""Page frame shared by every page: title, sidebar status, and the editable-case section."""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from . import api, case, results


def page(title: str, caption: str, icon: str = "📶"):
    st.set_page_config(page_title=f"{title} · Neighbour Impact", page_icon=icon, layout="wide")
    st.title(title)
    st.caption(caption)
    # sidebar_status()


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def hero(title: str, subtitle: str, image: str = "hero.jpg", height: str = "40vh", icon: str = "📶"):
    """Full-width banner with a background image behind the title (used on the Home page).

    image:  a file in ui/assets/ (jpg, png or webp). If it is missing, a plain colour band is shown instead.
    height: CSS height, e.g. "40vh" = 40% of the browser window height.
    """
    st.set_page_config(page_title=f"{title} · Neighbour Impact", page_icon=icon, layout="wide")
    path = ASSETS / image
    if path.exists():
        mime = {".png": "png", ".webp": "webp"}.get(path.suffix.lower(), "jpeg")
        data = base64.b64encode(path.read_bytes()).decode()
        background = f"url('data:image/{mime};base64,{data}') center / cover no-repeat"
    else:
        background = "#1F5AA6"
    st.markdown(f"""
<style>
  .hero {{
    height: {height}; min-height: 220px; border-radius: 12px; overflow: hidden; position: relative;
    background: {background}; display: flex; align-items: flex-end; margin-bottom: 1.5rem;
  }}
  .hero::before {{            /* dark gradient so white text stays readable on any photo */
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(10,20,35,0.10) 0%, rgba(10,20,35,0.70) 100%);
  }}
  .hero-text {{ position: relative; padding: 2rem 2.25rem; color: #FFFFFF; max-width: 60rem; }}
  .hero-text h1 {{ color: #FFFFFF; margin: 0 0 0.5rem 0; padding: 0; font-size: 2.6rem; line-height: 1.15; }}
  .hero-text p  {{ color: rgba(255,255,255,0.88); margin: 0; font-size: 1.05rem; }}
</style>
<div class="hero"><div class="hero-text"><h1>{title}</h1><p>{subtitle}</p></div></div>
""", unsafe_allow_html=True)
    # sidebar_status()


def sidebar_status():
    with st.sidebar:
        st.subheader("Service")
        r = api.ready()
        if r.get("status") != "ready":
            st.error(f"Model not ready: {r.get('error') or r.get('status')}")
            return
        try:
            info, data = api.model_info(), api.data_info()
        except api.ApiError as exc:
            st.error(exc.detail)
            return
        st.success("Model loaded")
        st.caption(f"{info.get('model')} · trained until {str(info.get('trained_until'))[:10]} · "
                   f"version {info.get('app_version')}")
        if data.get("loaded"):
            st.caption(f"Data store ({data['mode']} mode): {data['period_start'][:10]} to {data['period_end'][:10]}, "
                       f"{data['outages_listed']:,} outages listed")
        else:
            st.warning("No data store: only manual input and the API work.")


def case_section(state_key: str):
    """Editable inputs + Predict + results for the case kept in st.session_state[state_key]."""
    c = st.session_state.get(state_key)
    if not c:
        return
    lk = c["lookup"]
    real = lk.get("real_outage")
    used, total = len(lk["request"]["neighbours"]), lk["n_neighbours_total"]
    st.subheader(f"Outage of {lk['request']['failed_cell']['cell_id']}")
    st.write(f"Predicting the **{lk['prediction_time'][:16].replace('T', ' ')}** reading. "
             f"{used} of {total} neighbours were up and reporting just before the outage"
             + (f"; recorded outage lasted {real['duration_h']:.1f} h." if real and real.get("duration_h") else "."))
    if lk["skipped"]:
        with st.expander(f"{len(lk['skipped'])} neighbours left out"):
            st.dataframe(lk["skipped"], hide_index=True)

    with st.expander("Inputs: change any value, then predict", expanded=False):
        edited = case.editor(lk["request"], key=f"{state_key}_{c['version']}")
    changed = case.fingerprint(edited) != case.fingerprint(lk["request"])
    if changed:
        st.info("You changed the inputs, so this is a prediction for new values. Actual values are hidden "
                "because they belong to the original inputs.")
    if st.button("Predict neighbour impact", type="primary", key=f"{state_key}_go") or c.get("auto"):
        c["auto"] = False
        try:
            c["prediction"] = api.predict(edited)
            c["changed"] = changed
            c["request_used"] = edited
        except api.ApiError as exc:
            api.show_error(exc)
            c.pop("prediction", None)
    if c.get("prediction"):
        results.show(c["prediction"], None if c.get("changed") else lk["actual"], key=state_key)


def new_case(state_key: str, lookup: dict, auto_predict: bool = True):
    old = st.session_state.get(state_key) or {}
    st.session_state[state_key] = {"lookup": lookup, "version": old.get("version", 0) + 1, "auto": auto_predict}
