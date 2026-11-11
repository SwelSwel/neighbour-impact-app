"""Upload: a Dataset 2 file must give the same predictions as the notebook, and bad files must be rejected clearly.

The test file is rebuilt from the data store (KPI rows + OUTAGE/RECOVERY rows), so the test needs no raw data.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR, needs_data

pytestmark = needs_data
TARGETS = ["Latency", "Utilization", "Packet_Loss"]
RAW8 = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Alarm_Count", "Load", "Capacity"]


def dataset2_csv(case) -> bytes:
    """Dataset 2-format file for one outage: the failed cell, its neighbours and the other cells on its site."""
    kpi = pd.read_parquet(DATA_DIR / "kpi.parquet")
    nbrs = pd.read_parquet(DATA_DIR / "neighbours.parquet")
    cells = pd.read_parquet(DATA_DIR / "cells.parquet").set_index("cell_id")
    site = cells.at[case["cell_id"], "site_id"]
    keep = {case["cell_id"]} | set(nbrs.loc[nbrs.cell_id == case["cell_id"], "nbr_id"]) | \
           set(cells.index[cells.site_id == site])
    k = kpi[kpi.cell_id.astype(str).isin(keep)]
    rows = pd.DataFrame({"Timestamp": k.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"), "Cell_ID": k.cell_id.astype(str),
                         "Record_Type": "KPI", **{v: k[v] for v in RAW8}})
    o = pd.read_parquet(DATA_DIR / "outages.parquet")
    o = o[o.cell_id.isin(keep)]
    fmt = lambda s: pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S")
    ev = pd.concat([
        pd.DataFrame({"Timestamp": fmt(o.start), "Cell_ID": o.cell_id, "Record_Type": "OUTAGE",
                      "Cell_Down_Time": fmt(o.start)}),
        pd.DataFrame({"Timestamp": fmt(o.up), "Cell_ID": o.cell_id, "Record_Type": "RECOVERY",
                      "Cell_Up_Time": fmt(o.up)})[o.up.notna().to_numpy()]])
    return pd.concat([rows, ev], ignore_index=True).to_csv(index=False).encode()


@pytest.fixture(scope="module")
def upload_case(lookup_cases):
    c = lookup_cases[0]
    return c, dataset2_csv(c)


def test_upload_matches_notebook(client, auth, upload_case):
    c, content = upload_case
    r = client.post("/upload/predict-by-cell", headers=auth, files={"file": ("x.csv", content)},
                    data={"cell_id": c["cell_id"], "outage_start": c["outage_start"]})
    assert r.status_code == 200, r.text
    got = {n["cell_id"]: n for n in r.json()["prediction"]["neighbours"]}
    for nb, exp in c["neighbours"].items():
        for t in TARGETS:
            assert np.isclose(got[nb][t]["predicted"], exp["predicted"][t], atol=1e-3), (nb, t)


def test_upload_lists_outages_and_accepts_zip(client, auth, upload_case):
    c, content = upload_case
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readings.csv", content)
    r = client.post("/upload/outages", headers=auth, files={"file": ("x.zip", buf.getvalue())})
    assert r.status_code == 200, r.text
    assert c["cell_id"] in {o["cell_id"] for o in r.json()["outages"]}
    assert r.json()["report"]["kpi_rows"] > 0


@pytest.mark.parametrize("content, words", [
    (b"a,b\n1,2\n", "missing columns"),
    (b"Timestamp,Cell_ID,Record_Type,Traffic,Connected_Users,Throughput,Latency,Packet_Loss,Alarm_Count,Load,Capacity\n"
     b"2025-06-01 00:00:00,NOPE,KPI,1,1,1,1,1,1,1,1\n", "no usable KPI rows"),
    (b"PK\x03\x04 not really a zip", "could not read"),
])
def test_bad_uploads(client, auth, content, words):
    r = client.post("/upload/outages", headers=auth, files={"file": ("bad.csv", content)})
    assert r.status_code == 422 and words in r.json()["detail"], r.text


# ----------------------------------------------------------------------------------- new cells (Dataset 1 file)
def _as_new_cell(case, new_id="UG-NEW-0001-S01"):
    """The failed cell re-uploaded under a new ID: same site, same neighbour list, same readings and outage.

    The app must then give exactly the notebook's predictions, which proves new cells get the same treatment."""
    old = case["cell_id"]
    nbrs = pd.read_parquet(DATA_DIR / "neighbours.parquet")
    cells = pd.read_parquet(DATA_DIR / "cells.parquet").set_index("cell_id")
    listed = nbrs.loc[(nbrs.cell_id == old) & nbrs.is_listed, "nbr_id"].tolist()
    new_cells = pd.DataFrame([{"Site_ID": cells.at[old, "site_id"], "Cell_ID": new_id, "District": "X",
                               "Region": "X", "Technology": "4G", "Neighbor_Cell_ID": ",".join(listed)}])
    readings = pd.read_csv(io.BytesIO(dataset2_csv(case)), dtype=str)
    readings.loc[readings.Cell_ID == old, "Cell_ID"] = new_id
    return readings.to_csv(index=False).encode(), new_cells.to_csv(index=False).encode()


def test_new_cell_matches_notebook(client, auth, lookup_cases):
    c = lookup_cases[0]
    readings, new_cells = _as_new_cell(c)
    r = client.post("/upload/predict-by-cell", headers=auth,
                    files={"file": ("r.csv", readings), "cells": ("c.csv", new_cells)},
                    data={"cell_id": "UG-NEW-0001-S01", "outage_start": c["outage_start"]})
    assert r.status_code == 200, r.text
    got = {n["cell_id"]: n for n in r.json()["prediction"]["neighbours"]}
    assert len(got) == c["n_absorbing"]
    for nb, exp in c["neighbours"].items():
        for t in TARGETS:
            assert np.isclose(got[nb][t]["predicted"], exp["predicted"][t], atol=1e-3), (nb, t)


def test_new_cells_report_and_reverse_links(client, auth, lookup_cases):
    c = lookup_cases[0]
    readings, new_cells = _as_new_cell(c)
    for reverse, expect in [("true", True), ("false", False)]:
        r = client.post("/upload/outages", headers=auth, data={"add_reverse": reverse},
                        files={"file": ("r.csv", readings), "cells": ("c.csv", new_cells)})
        assert r.status_code == 200, r.text
        rep = r.json()["report"]["new_cells"]
        assert rep["new_cells"] == ["UG-NEW-0001-S01"] and rep["listed_links"] > 0
        assert (rep["reverse_links"] > 0) == expect


@pytest.mark.parametrize("cells_csv, words", [
    (b"Site_ID,Cell_ID\nS1,UG-NEW-1\n", "missing columns"),
    (b"Site_ID,Cell_ID,Neighbor_Cell_ID\nS1,UG-NEW-1,\nS1,UG-NEW-1,\n", "more than once"),
])
def test_bad_new_cells_file(client, auth, upload_case, cells_csv, words):
    _, content = upload_case
    r = client.post("/upload/outages", headers=auth, files={"file": ("r.csv", content), "cells": ("c.csv", cells_csv)})
    assert r.status_code == 422 and words in r.json()["detail"], r.text


def test_new_cell_id_already_exists(client, auth, upload_case):
    c, content = upload_case
    cells_csv = f"Site_ID,Cell_ID,Neighbor_Cell_ID\nS1,{c['cell_id']},\n".encode()
    r = client.post("/upload/outages", headers=auth, files={"file": ("r.csv", content), "cells": ("c.csv", cells_csv)})
    assert r.status_code == 422 and "already in Dataset 1" in r.json()["detail"]
