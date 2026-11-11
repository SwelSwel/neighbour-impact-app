"""API behaviour: normal use, end-to-end consistency, and failure scenarios."""
import copy

import numpy as np


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200


def test_auth_required(client):
    assert client.get("/model-info").status_code == 401
    assert client.post("/predict", json={}).status_code == 401


def test_model_info(client, auth):
    info = client.get("/model-info", headers=auth).json()
    assert info["n_features"] == 66 and set(info["targets"]) == {"Latency", "Utilization", "Packet_Loss"}


def test_example_round_trip_matches_notebook(client, auth, cases):
    """Request built from a real test outage -> /predict -> same predictions as the notebook for that neighbour."""
    for idx in range(5):
        body = client.get(f"/example?index={idx}", headers=auth).json()
        r = client.post("/predict", json=body, headers=auth)
        assert r.status_code == 200, r.text
        first = r.json()["neighbours"][0]
        for t in ["Latency", "Utilization", "Packet_Loss"]:
            assert np.isclose(first[t]["predicted"], cases[idx]["expected_predictions"][t], rtol=1e-4, atol=1e-3)


def test_response_shape(client, auth):
    body = client.get("/example", headers=auth).json()
    out = client.post("/predict", json=body, headers=auth).json()
    assert len(out["neighbours"]) == len(body["neighbours"])
    for n in out["neighbours"]:
        for t in ["Latency", "Utilization", "Packet_Loss"]:
            f = n[t]
            assert f["range_80_low"] <= f["predicted"] <= f["range_80_high"]
            assert f["severe"] <= f["degraded"]          # severe implies degraded
    assert "X-Request-ID" in client.post("/predict", json=body, headers=auth).headers


def _example(client, auth):
    return client.get("/example", headers=auth).json()


def test_rejects_zero_capacity(client, auth):
    body = _example(client, auth); body["neighbours"][0]["now"]["Capacity"] = 0
    r = client.post("/predict", json=body, headers=auth)
    assert r.status_code == 422 and any("Capacity" in e["field"] for e in r.json()["errors"])


def test_rejects_negative_kpi_and_bad_packet_loss(client, auth):
    body = _example(client, auth); body["neighbours"][0]["now"]["Latency"] = -5
    assert client.post("/predict", json=body, headers=auth).status_code == 422
    body = _example(client, auth); body["neighbours"][0]["h3_before"]["Packet_Loss"] = 150
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_rejects_unknown_field(client, auth):
    body = _example(client, auth); body["failed_cell"]["Colour"] = "red"
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_missing_values_are_allowed(client, auth):
    """A blank KPI is a missing value, as in training: the model still predicts."""
    body = _example(client, auth)
    del body["failed_cell"]["Load"]
    body["neighbours"][0]["h3_before"] = None
    body["neighbours"][0]["now"]["Throughput"] = None
    r = client.post("/predict", json=body, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["neighbours"][0]["load_ratio"] is None      # needs the failed cell's load


def test_too_many_neighbours(client, auth):
    from app.config import settings
    body = _example(client, auth)
    n0 = body["neighbours"][0]
    body["neighbours"] = [dict(n0, cell_id=f"N{i}") for i in range(settings.max_neighbours + 1)]
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_rejects_empty_and_duplicate_neighbours(client, auth):
    body = _example(client, auth); body["neighbours"] = []
    assert client.post("/predict", json=body, headers=auth).status_code == 422
    body = _example(client, auth); body["neighbours"].append(copy.deepcopy(body["neighbours"][0]))
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_rejects_failed_cell_as_neighbour(client, auth):
    body = _example(client, auth); body["neighbours"][0]["cell_id"] = body["failed_cell"]["cell_id"]
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_rejects_bad_timestamp(client, auth):
    body = _example(client, auth); body["outage_start"] = "yesterday"
    assert client.post("/predict", json=body, headers=auth).status_code == 422


def test_single_neighbour_without_last_week(client, auth):
    body = _example(client, auth); body["neighbours"] = body["neighbours"][:1]; body["neighbours"][0]["last_week"] = None
    r = client.post("/predict", json=body, headers=auth)
    assert r.status_code == 200 and len(r.json()["neighbours"]) == 1
