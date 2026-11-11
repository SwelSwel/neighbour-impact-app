"""Turn "cell X went down at time T" into a full prediction request, exactly as notebook section 8 builds a row.

For an outage of cell c starting at T:
  se  = first 3-hourly slot at or after T           (the reading the model predicts)
  j   = se - 1                                      (the last reading before the outage)
  neighbours that count = linked cells with a reading at j that were not down at j
  lookback = the neighbour's readings at j - 1 and j - 2; last week = slot se - one week (if up and reporting)
  failed cell = its reading at j; site_cells_down = other cells on its site already down at T
  actual = the neighbour's reading at se, if it reported and was not down (shown next to the prediction)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .kpi_store import KPIStore
from .outages import site_cells_down
from .topology import Topology

FAILED_KPIS = ["Traffic", "Connected_Users", "Load", "Capacity", "Throughput"]
LAST_WEEK = ["Latency", "Utilization", "Packet_Loss", "Traffic", "Load"]
TARGETS = ["Latency", "Utilization", "Packet_Loss"]


class LookupFailed(Exception):
    """A lookup that cannot be done; `status` is the HTTP status the API returns."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _util(v: dict) -> float | None:
    load, cap = v.get("Load"), v.get("Capacity")
    return 100 * load / cap if load is not None and cap else None


def lookup(store: KPIStore, topo: Topology, cell_id: str, outage_start: datetime,
           require_real_outage: bool = False) -> dict:
    """Returns a dict matching schemas.LookupResponse. Raises LookupFailed with a clear message."""
    cell_id = cell_id.strip()
    if not topo.has(cell_id):
        raise LookupFailed(404, f"Cell {cell_id} is not in Dataset 1 (the cell list).")
    se = store.grid.slot_ceil(outage_start)
    j = se - 1
    if not (store.covers(j - 2) and store.covers(j)):
        a, b = store.period
        raise LookupFailed(422, f"No stored readings for {outage_start:%Y-%m-%d %H:%M}. "
                                f"The {store.source} covers {a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M}, and a lookup "
                                f"needs the 6 hours before the outage.")
    real = store.find_outage(cell_id, outage_start)
    if require_real_outage and real is None:
        raise LookupFailed(422, f"Cell {cell_id} has no recorded outage at {outage_start:%Y-%m-%d %H:%M}. This data "
                                f"store only keeps the readings around real outages: pick one from the outage list, "
                                f"or export the store with STORE_MODE = 'full' to look up any cell and time.")

    links = topo.neighbours(cell_id)
    if not links:
        raise LookupFailed(422, f"Cell {cell_id} has no neighbours in Dataset 1, so there is nothing to predict.")

    neighbours, skipped, actual = [], [], {}
    for nb in links:
        now = store.reading(nb.cell_id, j)
        if now is None:
            skipped.append({"cell_id": nb.cell_id, "reason": "no reading just before the outage"})
            continue
        if now.down:
            skipped.append({"cell_id": nb.cell_id, "reason": "down just before the outage"})
            continue
        h3, h6 = store.reading(nb.cell_id, j - 1), store.reading(nb.cell_id, j - 2)
        lw = store.reading(nb.cell_id, se - store.week_slots)
        last_week = None
        if lw is not None and not lw.down:
            v = dict(lw.values, Utilization=_util(lw.values))
            last_week = {k: v.get(k) for k in LAST_WEEK}
        at = store.reading(nb.cell_id, se)
        actual[nb.cell_id] = (None if at is None or at.down else
                              {"Latency": at.values["Latency"], "Utilization": _util(at.values),
                               "Packet_Loss": at.values["Packet_Loss"]})
        neighbours.append({"cell_id": nb.cell_id, "is_same_site": nb.is_same_site, "is_listed": nb.is_listed,
                           "now": now.values, "h3_before": h3.values if h3 else None,
                           "h6_before": h6.values if h6 else None, "last_week": last_week})
    if not neighbours:
        raise LookupFailed(422, f"None of the {len(links)} neighbours of {cell_id} was up and reporting just before "
                                f"the outage, so no neighbour can take over its users.")

    failed = store.reading(cell_id, j)
    fvals = {k: (failed.values.get(k) if failed else None) for k in FAILED_KPIS}
    t0 = real["start"] if real else outage_start
    site = topo.site_of(cell_id)
    request = {"outage_start": t0, "failed_cell": {"cell_id": cell_id, **fvals,
               "site_cells_down": site_cells_down(store.outages, topo.cells_on_site(site), cell_id, t0)},
               "neighbours": neighbours}
    return {"request": request, "prediction_time": store.grid.time(se), "actual": actual, "real_outage": real,
            "n_neighbours_total": len(links), "skipped": skipped, "source": store.source}
