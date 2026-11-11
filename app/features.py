"""Feature construction shared by training and production.

This is the same logic as `impact_features()` in Neighbour_Impact_Prediction.ipynb (section 19). The automated
tests rebuild the notebook's own test rows through this module and require an exact match, so any change here that
breaks training/production consistency fails CI before it can be deployed.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

# Defaults match the notebook; the predictor overrides them with the lists stored in the model bundle.
RAW8 = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Alarm_Count", "Load", "Capacity"]
VARS9 = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Alarm_Count", "Load", "Capacity", "Utilization"]
TARGETS = ["Latency", "Utilization", "Packet_Loss"]
NORMAL_VARS = TARGETS + ["Traffic", "Load"]
FAIL_VARS = ["Traffic", "Connected_Users", "Load", "Capacity", "Utilization", "Throughput"]


def utilization(reading: Mapping[str, float]) -> float:
    """Load / Capacity x 100, or NaN when capacity is missing or not positive (as in the notebook)."""
    cap = reading.get("Capacity", np.nan)
    return 100 * reading.get("Load", np.nan) / cap if cap and cap > 0 else np.nan


def impact_features(nb_now: Mapping[str, float], nb_3h: Mapping[str, float], nb_6h: Mapping[str, float],
                    failed: Mapping[str, float], n_absorbing: int, total_absorbing_capacity: float, hour: int,
                    is_same_site: int, is_listed: int, site_cells_down: int = 0,
                    nb_last_week: Mapping[str, float] | None = None,
                    vars9: list[str] = VARS9, normal_vars: list[str] = NORMAL_VARS,
                    fail_vars: list[str] = FAIL_VARS) -> dict[str, float]:
    """Build one feature row from values known before the outage.

    nb_now, nb_3h, nb_6h: the neighbour's 8 raw KPIs at its last reading before the outage, 3 h and 6 h earlier.
    failed: Traffic, Connected_Users, Load, Capacity, Throughput of the failed cell at its last reading.
    nb_last_week: optional Latency, Utilization, Packet_Loss, Traffic, Load at the same slot one week earlier.
    """
    obs = [dict(d, Utilization=utilization(d)) for d in (nb_now, nb_3h, nb_6h)]
    f: dict[str, float] = {}
    for v in vars9:
        cur, l1, l2 = (float(o.get(v, np.nan)) for o in obs)
        f[f"nb_{v}"] = cur
        f[f"nb_{v}_lag1"] = l1
        f[f"nb_{v}_lag2"] = l2
        f[f"nb_{v}_mean2"] = (l1 + l2) / 2
        f[f"nb_{v}_trend2"] = cur - l2
    lwk = nb_last_week or {}
    for v in normal_vars:
        val = lwk.get(v)
        f[f"nb_{v}_last_week"] = float("nan") if val is None else float(val)
    fl = dict(failed, Utilization=utilization(failed))
    for v in fail_vars:
        f[f"fail_{v}"] = float(fl.get(v, np.nan))
    capn = obs[0].get("Capacity", np.nan)
    share = capn / total_absorbing_capacity if total_absorbing_capacity > 0 else np.nan
    f.update({
        "n_absorbing": n_absorbing,
        "capacity_share": share,
        "expected_extra_load": f["fail_Load"] * share,
        "load_ratio": f["fail_Load"] / capn if capn > 0 else np.nan,
        "hour": hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "is_same_site": int(is_same_site),
        "is_listed": int(is_listed),
        "site_cells_down": site_cells_down,
    })
    return f
