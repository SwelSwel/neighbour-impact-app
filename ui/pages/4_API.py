"""Page 4: the API for other systems, with copy-ready examples and a request tester."""
import json

import streamlit as st

from lib import api
from lib.layout import page
from lib.settings import PUBLIC_API_URL

page("4 · API", "Other systems can call the same predictions over HTTP. Every call except /health and /ready needs "
                "the API key in the X-API-Key header.")

st.markdown(f"Base URL: `{PUBLIC_API_URL}` · interactive documentation: [{PUBLIC_API_URL}/docs]({PUBLIC_API_URL}/docs)")
st.markdown("""
| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health`, `/ready` | – | Liveness; model and data store loaded |
| GET | `/model-info`, `/data-info` | – | Model versions and test metrics; what the data store holds |
| GET | `/outages` | `split`, `q`, `technology`, `region`, `limit` | Real outages in the data store |
| GET | `/cells?q=` · `/cells/{id}` | text | Cell search · one cell and its neighbours |
| POST | `/predict-by-cell` | `{"cell_id", "outage_start"}` | Lookup + predictions + actual values |
| POST | `/lookup` | `{"cell_id", "outage_start"}` | The full input, to edit and send to `/predict` |
| POST | `/predict` | Full input: failed cell + every neighbour's readings | Predictions for any values |
| POST | `/upload/outages`, `/upload/lookup`, `/upload/predict-by-cell` | Dataset 2 file (multipart) | As above, from the file |
""")

case = st.session_state.get("case_p1")
if case:   # the outage chosen on page 1
    cell, start = case["lookup"]["request"]["failed_cell"]["cell_id"], case["lookup"]["request"]["outage_start"]
else:      # otherwise the first real outage in the data store
    try:
        first = api.outages()[0]
        cell, start = first["cell_id"], first["start"]
    except (api.ApiError, IndexError):
        cell, start = "<cell id>", "2025-06-15T10:40:00"
st.subheader("Examples")
st.code(f"""curl -X POST {PUBLIC_API_URL}/predict-by-cell \\
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \\
  -d '{{"cell_id": "{cell}", "outage_start": "{start}"}}'""", language="bash")
st.code(f"""import requests
r = requests.post("{PUBLIC_API_URL}/predict-by-cell", headers={{"X-API-Key": API_KEY}},
                  json={{"cell_id": "{cell}", "outage_start": "{start}"}})
for n in r.json()["prediction"]["neighbours"]:
    print(n["cell_id"], n["Latency"]["predicted"], n["Latency"]["degraded"])""", language="python")

st.subheader("Try a request")
endpoints = {"POST /predict-by-cell": ("POST", "/predict-by-cell"), "POST /lookup": ("POST", "/lookup"),
             "POST /predict": ("POST", "/predict"), "GET /outages": ("GET", "/outages?limit=5"),
             "GET /model-info": ("GET", "/model-info"), "GET /data-info": ("GET", "/data-info"),
             "GET /ready": ("GET", "/ready")}
choice = st.selectbox("Endpoint", list(endpoints))
method, path = endpoints[choice]
if path == "/predict":
    default = case["lookup"]["request"] if case else None
    if default is None:
        try:
            default = api.call("GET", "/example")
        except api.ApiError:
            default = {}
else:
    default = {"cell_id": cell, "outage_start": start}
body = st.text_area("JSON body", json.dumps(default, indent=1, default=str), height=260,
                    disabled=method == "GET", key=f"body_{choice}")
if st.button("Send", type="primary"):
    try:
        out = api.call(method, path, json=json.loads(body)) if method == "POST" else api.call(method, path)
        st.success("200 OK")
        st.json(out, expanded=2)
    except json.JSONDecodeError as exc:
        st.error(f"The body is not valid JSON: {exc}")
    except api.ApiError as exc:
        st.error(f"{exc.status}: {exc.detail}")
        if exc.errors:
            st.dataframe(exc.errors, hide_index=True)
