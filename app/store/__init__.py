"""Data access for the app: the cell topology (Dataset 1) and KPI readings (Dataset 2), stored or uploaded."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .kpi_store import KPIStore
from .topology import Topology


@dataclass
class DataStore:
    topology: Topology
    kpi: KPIStore

    @classmethod
    def from_dir(cls, path: Path) -> "DataStore":
        return cls(Topology.from_dir(path), KPIStore.from_dir(path))

    @property
    def mode(self) -> str:
        return self.kpi.meta.get("mode", "full")

    def info(self) -> dict:
        a, b = self.kpi.period
        o = self.kpi.outages
        return {"mode": self.mode, "period_start": a, "period_end": b,
                "test_period_start": self.kpi.meta.get("test_period_start"),
                "cells": int(len(self.topology.cells)), "neighbour_links": int(len(self.topology.links)),
                "kpi_rows": int(self.kpi.n_rows),
                "outages_listed": int((o.get("split", pd.Series(dtype=str)) != "").sum()) if len(o) else 0,
                "created_utc": self.kpi.meta.get("created_utc")}
