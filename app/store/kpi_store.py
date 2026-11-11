"""3-hourly KPI readings, indexed by (cell, slot) for fast lookup, plus the outage list of the same period.

Two sources produce a KPIStore:
  * the data store exported by notebook section 19c (`from_dir`)
  * a file the user uploads in Dataset 2 format (`app.store.dataset2.store_from_upload`)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import Grid

RAW8 = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Alarm_Count", "Load", "Capacity"]
_OFFSET = 2 ** 31


@dataclass
class Reading:
    values: dict[str, float | None]        # the 8 raw KPIs; None = not recorded
    down: bool                             # the cell was in an outage at this slot


@dataclass
class KPIStore:
    readings: pd.DataFrame | None          # cell_id, slot, RAW8..., down (released after indexing)
    grid: Grid
    outages: pd.DataFrame                  # cell_id, start, up, duration_h (+ event_id, split, ... when exported)
    meta: dict = field(default_factory=dict)
    source: str = "data store"

    def __post_init__(self):
        r = self.readings
        cell = r["cell_id"] if isinstance(r["cell_id"].dtype, pd.CategoricalDtype) else r["cell_id"].astype("category")
        self._code = {str(c): i for i, c in enumerate(cell.cat.categories)}
        self.cells = sorted(self._code)
        keys = cell.cat.codes.to_numpy(np.int64) << 32 | (r["slot"].to_numpy(np.int64) + _OFFSET)
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._vals = r[RAW8].to_numpy(np.float32)[order]           # float32, as stored and as used in training
        self._down = r["down"].to_numpy(bool)[order]
        self.n_rows = len(r)
        self.slot_min = int(r["slot"].min()) if len(r) else 0
        self.slot_max = int(r["slot"].max()) if len(r) else -1
        self.readings = None                                        # arrays above hold everything; free the table
        o = self.outages
        if len(o):
            o["start"] = pd.to_datetime(o["start"])
            o["up"] = pd.to_datetime(o["up"])
            o["slot"] = [self.grid.slot_ceil(t) for t in o["start"]]
        else:
            self.outages = pd.DataFrame(columns=["cell_id", "start", "up", "duration_h", "slot"])

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dir(cls, path: Path) -> "KPIStore":
        meta = json.loads((path / "meta.json").read_text())
        grid = Grid(pd.Timestamp(meta["grid_start"]).to_pydatetime(), int(meta["grid_step_h"]))
        readings = pd.read_parquet(path / "kpi.parquet", columns=["cell_id", "slot", *RAW8, "down"])
        outages = pd.read_parquet(path / "outages.parquet")
        return cls(readings, grid, outages, meta, "data store")

    # ---------------------------------------------------------------- queries
    def reading(self, cell_id: str, slot: int) -> Reading | None:
        code = self._code.get(cell_id)
        if code is None:
            return None
        key = np.int64(code) << 32 | np.int64(slot + _OFFSET)
        i = int(np.searchsorted(self._keys, key))
        if i >= len(self._keys) or self._keys[i] != key:
            return None
        v = self._vals[i]
        return Reading({k: (None if np.isnan(x) else float(x)) for k, x in zip(RAW8, v.tolist())}, bool(self._down[i]))

    def covers(self, slot: int) -> bool:
        return self.slot_min <= slot <= self.slot_max

    @property
    def period(self) -> tuple:
        return self.grid.time(self.slot_min), self.grid.time(self.slot_max)

    @property
    def week_slots(self) -> int:
        return int(self.meta.get("week_slots", 7 * 24 // self.grid.step_h))

    def find_outage(self, cell_id: str, start) -> dict | None:
        """The recorded outage of this cell whose first predicted slot is the same as for `start`."""
        o = self.outages
        if o.empty:
            return None
        m = o[(o.cell_id == cell_id) & (o.slot == self.grid.slot_ceil(start))]
        if m.empty:
            return None
        r = m.iloc[(m.start - pd.Timestamp(start)).abs().argmin()]
        return {k: (None if pd.isna(v) else (v.to_pydatetime() if isinstance(v, pd.Timestamp) else
                                             (v.item() if hasattr(v, "item") else v)))
                for k, v in r.items() if k != "slot"}
