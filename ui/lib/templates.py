"""Downloadable CSV templates for the upload page, with a few example rows to replace.

Readings template  = Dataset 2 format (the monthly files): one row per 3-hourly KPI reading, plus OUTAGE / RECOVERY rows.
New-cells template = Dataset 1 format (the cell information file): one row per new cell.
"""
from __future__ import annotations

import pandas as pd

READING_COLUMNS = ["Timestamp", "Site_ID", "Cell_ID", "Record_Type", "Status", "Traffic", "Connected_Users",
                   "Throughput", "Latency", "Packet_Loss", "Load", "Capacity", "Alarm_Count", "Previous_Outage_Count",
                   "Cell_Down_Time", "Cell_Up_Time"]
CELL_COLUMNS = ["Site_ID", "Cell_ID", "District", "Region", "Technology", "Neighbor_Cell_ID"]

# Columns the app reads, and what they mean (shown on the page)
READING_HELP = pd.DataFrame([
    ("Timestamp", "Yes", "YYYY-MM-DD HH:MM:SS. KPI readings on the 3-hourly grid: 00:00, 03:00 … 21:00"),
    ("Cell_ID", "Yes", "The cell the row belongs to"),
    ("Record_Type", "Yes", "KPI for a reading; OUTAGE or RECOVERY for an event"),
    ("Traffic, Connected_Users, Throughput, Latency (ms), Packet_Loss (%), Load, Capacity, Alarm_Count",
     "Yes (KPI rows)", "The 8 KPIs. Leave blank on OUTAGE / RECOVERY rows. Utilization is calculated (100 × Load ÷ Capacity)"),
    ("Cell_Down_Time / Cell_Up_Time", "Optional", "On OUTAGE / RECOVERY rows: when the cell went down / came back"),
    ("Site_ID, Status, Previous_Outage_Count", "Optional", "Not used by the model"),
], columns=["Column", "Required", "Meaning"])

CELL_HELP = pd.DataFrame([
    ("Site_ID", "Yes", "The tower the cell is on. An existing Site_ID makes the cells on that tower same-site neighbours"),
    ("Cell_ID", "Yes", "A new, unique ID"),
    ("Neighbor_Cell_ID", "Yes", "Neighbour cell IDs separated by commas, in quotes. Existing or other new cells"),
    ("Technology, Region, District", "Optional", "Not used by the model"),
], columns=["Column", "Required", "Meaning"])


def _kpi(ts, site, cell, traffic, users, thr, lat, pl, load, cap):
    return {"Timestamp": ts, "Site_ID": site, "Cell_ID": cell, "Record_Type": "KPI", "Status": "UP",
            "Traffic": traffic, "Connected_Users": users, "Throughput": thr, "Latency": lat, "Packet_Loss": pl,
            "Load": load, "Capacity": cap, "Alarm_Count": 0, "Previous_Outage_Count": 0}


def readings_template(failed: str, neighbour: str) -> str:
    """Example: `failed` goes down at 22:10 on 14 June; `neighbour` is one of its neighbours."""
    site = lambda c: c.rsplit("-", 1)[0]
    rows = []
    for ts, k in [("2025-06-14 15:00:00", 1.00), ("2025-06-14 18:00:00", 1.15), ("2025-06-14 21:00:00", 1.10)]:
        rows.append(_kpi(ts, site(failed), failed, round(12.4 * k, 3), round(85 * k), round(9.8 * k, 3),
                         round(48.2 / k, 3), 0.92, round(120.5 * k, 3), 150.0))
        rows.append(_kpi(ts, site(neighbour), neighbour, round(9.1 * k, 3), round(64 * k), round(7.5 * k, 3),
                         round(52.7 / k, 3), 1.05, round(88.3 * k, 3), 140.0))
    rows.append({"Timestamp": "2025-06-14 22:10:00", "Site_ID": site(failed), "Cell_ID": failed,
                 "Record_Type": "OUTAGE", "Status": "DOWN", "Cell_Down_Time": "2025-06-14 22:10:00"})
    rows.append(_kpi("2025-06-15 00:00:00", site(neighbour), neighbour, 13.6, 98, 9.9, 81.4, 1.62, 131.0, 140.0))
    rows.append({"Timestamp": "2025-06-15 01:40:00", "Site_ID": site(failed), "Cell_ID": failed,
                 "Record_Type": "RECOVERY", "Status": "UP", "Cell_Down_Time": "2025-06-14 22:10:00",
                 "Cell_Up_Time": "2025-06-15 01:40:00"})
    df = pd.DataFrame(rows, columns=READING_COLUMNS)
    for c in ["Connected_Users", "Alarm_Count", "Previous_Outage_Count"]:
        df[c] = df[c].astype("Int64")                    # whole numbers, blank on event rows
    return df.to_csv(index=False)


def new_cells_template(existing: list[str]) -> str:
    """Example: a new tower UG-NEW-0001 with two cells, each listing the other and some existing cells."""
    ex = (existing + ["<existing cell ID>"] * 3)[:3]
    rows = [{"Site_ID": "UG-NEW-0001", "Cell_ID": "UG-NEW-0001-S01", "District": "Kampala", "Region": "Central",
             "Technology": "4G", "Neighbor_Cell_ID": ",".join(["UG-NEW-0001-S02", ex[0], ex[1]])},
            {"Site_ID": "UG-NEW-0001", "Cell_ID": "UG-NEW-0001-S02", "District": "Kampala", "Region": "Central",
             "Technology": "4G", "Neighbor_Cell_ID": ",".join(["UG-NEW-0001-S01", ex[1], ex[2]])}]
    return pd.DataFrame(rows, columns=CELL_COLUMNS).to_csv(index=False)
