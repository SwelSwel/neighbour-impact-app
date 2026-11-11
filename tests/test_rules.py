"""The data rules copied from the notebook: time grid, timestamps, outage pairing, "down" slots, site counts."""
from datetime import datetime

import numpy as np
import pandas as pd

from app.store.dataset2 import parse_ts
from app.store.grid import Grid
from app.store.outages import down_slots, pair_events, site_cells_down

G = Grid(datetime(2025, 1, 1), 3)
T = pd.Timestamp


def test_slot_ceil():
    assert G.slot_ceil(datetime(2025, 1, 1, 0, 0)) == 0
    assert G.slot_ceil(datetime(2025, 1, 1, 0, 1)) == 1
    assert G.slot_ceil(datetime(2025, 1, 2, 10, 40)) == 8 + 4      # next reading 12:00
    assert G.exact_slot(datetime(2025, 1, 1, 3)) == 1 and G.exact_slot(datetime(2025, 1, 1, 4)) is None


def test_parse_ts_bare_date_is_midnight():
    s = parse_ts(pd.Series(["2025-06-01", "2025-06-01 03:00:00", "garbage"]))
    assert s[0] == T("2025-06-01 00:00:00") and s[1] == T("2025-06-01 03:00:00") and pd.isna(s[2])


def _events(rows):
    return pd.DataFrame(rows, columns=["Cell_ID", "Record_Type", "Timestamp", "Cell_Down_Time", "Cell_Up_Time"])


def test_pairing_and_recovery_of_a_later_outage():
    ev = _events([("A", "OUTAGE", T("2025-01-01 01:00"), T("2025-01-01 01:00"), pd.NaT),
                  ("A", "OUTAGE", T("2025-01-01 10:00"), T("2025-01-01 10:00"), pd.NaT),
                  ("A", "RECOVERY", T("2025-01-01 12:00"), pd.NaT, T("2025-01-01 12:00"))])
    o = pair_events(ev)
    assert len(o) == 2
    assert pd.isna(o.up.iloc[0])                         # the only recovery comes after the next outage
    assert o.up.iloc[1] == T("2025-01-01 12:00") and o.duration_h.iloc[1] == 2


def test_down_slots_with_and_without_recovery():
    o = pd.DataFrame({"cell_id": ["A", "B"], "start": [T("2025-01-01 01:00"), T("2025-01-01 01:00")],
                      "up": [T("2025-01-01 07:00"), pd.NaT]})
    d = down_slots(o, G, {"B": np.array([0, 3, 4])}, last_slot=10)
    assert {s for c, s in d if c == "A"} == {1, 2}       # from ceil(start)=1 up to ceil(up)=3, exclusive
    assert {s for c, s in d if c == "B"} == {1, 2}       # no recovery: until the next reading (slot 3)


def test_site_cells_down():
    o = pd.DataFrame({"cell_id": ["B", "C", "D"], "start": [T("2025-01-01 01:00")] * 3,
                      "up": [T("2025-01-01 05:00"), T("2025-01-01 02:00"), pd.NaT]})
    assert site_cells_down(o, {"A", "B", "C"}, "A", T("2025-01-01 03:00")) == 1     # B yes, C recovered, D other site
