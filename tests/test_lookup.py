"""Cell + time lookup: the app must rebuild the notebook's inputs and predictions for real test outages."""
import numpy as np
import pytest

from conftest import needs_data

pytestmark = needs_data
TARGETS = ["Latency", "Utilization", "Packet_Loss"]


def _predict_by_cell(client, auth, cell, start):
    return client.post("/predict-by-cell", headers=auth, json={"cell_id": cell, "outage_start": start})


def test_predictions_match_notebook(client, auth, lookup_cases):
    """Every neighbour of every exported test outage: same prediction and same actual value as the notebook."""
    assert lookup_cases, "lookup_test_cases.json is empty"
    for c in lookup_cases:
        r = _predict_by_cell(client, auth, c["cell_id"], c["outage_start"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["lookup"]["request"]["neighbours"]) == c["n_absorbing"], "different set of absorbing neighbours"
        got = {n["cell_id"]: n for n in body["prediction"]["neighbours"]}
        for nb, exp in c["neighbours"].items():
            for t in TARGETS:
                assert np.isclose(got[nb][t]["predicted"], exp["predicted"][t], atol=1e-3), (c["cell_id"], nb, t)
                assert np.isclose(got[nb][t]["actual"], exp["actual"][t], atol=1e-3), (c["cell_id"], nb, t)


def test_lookup_then_predict_is_identical(client, auth, lookup_cases):
    """POST /lookup + POST /predict (the website's path) gives the same result as /predict-by-cell."""
    c = lookup_cases[0]
    lk = client.post("/lookup", headers=auth, json={"cell_id": c["cell_id"], "outage_start": c["outage_start"]}).json()
    a = client.post("/predict", headers=auth, json=lk["request"]).json()
    b = _predict_by_cell(client, auth, c["cell_id"], c["outage_start"]).json()["prediction"]
    assert [n["Latency"]["predicted"] for n in a["neighbours"]] == [n["Latency"]["predicted"] for n in b["neighbours"]]


def test_outage_list(client, auth):
    rows = client.get("/outages", headers=auth).json()
    assert rows and all(r["split"] in {"test", "unseen site"} for r in rows)
    assert all(r["split"] == "test" for r in client.get("/outages?split=test", headers=auth).json())


def test_cell_search_and_detail(client, auth, lookup_cases):
    cid = lookup_cases[0]["cell_id"]
    assert cid in client.get(f"/cells?q={cid}", headers=auth).json()["cells"]
    d = client.get(f"/cells/{cid}", headers=auth).json()
    assert d["cell_id"] == cid and d["neighbours"]


@pytest.mark.parametrize("cell, start, status, words", [
    ("NO-SUCH-CELL", "2025-06-15T10:00:00", 404, "not in Dataset 1"),
    (None, "2020-01-01T10:00:00", 422, "No stored readings"),
])
def test_lookup_failures(client, auth, lookup_cases, cell, start, status, words):
    r = client.post("/lookup", headers=auth, json={"cell_id": cell or lookup_cases[0]["cell_id"], "outage_start": start})
    assert r.status_code == status and words in r.json()["detail"]


def test_not_a_real_outage_in_outages_mode(client, auth, lookup_cases):
    from app.config import settings
    if client.get("/data-info", headers=auth).json()["mode"] != "outages" or settings.allow_any_time:
        pytest.skip("only applies to an 'outages' data store")
    c = lookup_cases[0]
    start = np.datetime64(c["outage_start"][:19]) - np.timedelta64(2, "D")
    r = client.post("/lookup", headers=auth, json={"cell_id": c["cell_id"], "outage_start": str(start)})
    assert r.status_code == 422 and "no recorded outage" in r.json()["detail"]


def test_bad_lookup_body(client, auth):
    assert client.post("/lookup", headers=auth, json={"cell_id": "X"}).status_code == 422
    assert client.post("/lookup", headers=auth, json={"cell_id": "X", "outage_start": "soon"}).status_code == 422
