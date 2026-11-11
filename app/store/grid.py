"""The 3-hourly time grid, identical to the notebook's (slot 0 = GRID_START, one slot every 3 hours)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass(frozen=True)
class Grid:
    start: datetime
    step_h: int = 3

    @property
    def step(self) -> timedelta:
        return timedelta(hours=self.step_h)

    def slot_ceil(self, ts: datetime) -> int:
        """First slot at or after ts. The notebook's `se` for an outage starting at ts."""
        return math.ceil((pd.Timestamp(ts) - pd.Timestamp(self.start)) / self.step)

    def exact_slot(self, ts) -> int | None:
        """Slot index if ts lies exactly on the grid, else None (off-grid KPI rows are ignored, as in the notebook)."""
        off = (pd.Timestamp(ts) - pd.Timestamp(self.start)) / self.step
        return int(off) if off == int(off) else None

    def time(self, slot: int) -> datetime:
        return (pd.Timestamp(self.start) + slot * self.step).to_pydatetime()
