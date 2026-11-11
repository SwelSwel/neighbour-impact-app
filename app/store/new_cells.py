"""New cells uploaded by the user in Dataset 1 format, added to the stored topology for one request only.

Rules (the same as notebook section 2, applied to the new rows):
  * Neighbor_Cell_ID is split on "," or ";"; the cell itself and IDs that exist nowhere are ignored (and reported).
  * Every other cell on the same Site_ID is a same-site neighbour, in both directions.
  * Optionally (add_reverse=True) a new cell is also added to the lists of the cells it names, so that it can take
    users when one of those existing cells fails. Real networks set neighbour relations up in both directions.
Nothing is saved: the stored Dataset 1 is never changed.
"""
from __future__ import annotations

import copy
import io

import pandas as pd

from .dataset2 import UploadError
from .topology import Neighbour, Topology

REQUIRED = ["Site_ID", "Cell_ID", "Neighbor_Cell_ID"]
OPTIONAL = ["Technology", "Region", "District"]


def read_cells_table(content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise UploadError(f"could not read the new-cells file as CSV: {exc}") from exc
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise UploadError(f"new-cells file: missing columns {', '.join(missing)} (expected the Dataset 1 format)")
    for c in REQUIRED + [c for c in OPTIONAL if c in df.columns]:
        df[c] = df[c].astype(str).str.strip()
    df = df[df.Cell_ID != ""]
    if df.empty:
        raise UploadError("new-cells file: no rows with a Cell_ID")
    if (df.Site_ID == "").any():
        raise UploadError("new-cells file: every new cell needs a Site_ID")
    dup = df.Cell_ID[df.Cell_ID.duplicated()].unique().tolist()
    if dup:
        raise UploadError(f"new-cells file: Cell_ID listed more than once: {', '.join(dup[:10])}")
    return df


def add_new_cells(topo: Topology, df: pd.DataFrame, add_reverse: bool = True) -> tuple[Topology, dict]:
    """Return a copy of `topo` with the new cells and their links, plus a short report."""
    clash = [c for c in df.Cell_ID if topo.has(c)]
    if clash:
        raise UploadError(f"these cells are already in Dataset 1: {', '.join(clash[:10])}. Upload only their "
                          f"readings (first option), or give the new cells new IDs.")
    new_ids = set(df.Cell_ID)
    site = dict(zip(df.Cell_ID, df.Site_ID))
    site_of = lambda c: site[c] if c in site else topo.site_of(c)
    exists = lambda c: c in new_ids or topo.has(c)

    t = copy.copy(topo)                                  # shallow copy: only the changed entries are replaced
    t._nbrs = dict(topo._nbrs)
    t._site_cells = {k: set(v) for k, v in topo._site_cells.items()}
    links: dict[str, dict[str, Neighbour]] = {}          # cells whose lists change: cell -> {nbr_id: Neighbour}

    def link(a: str, b: str, listed: bool):
        if a not in links:
            links[a] = {n.cell_id: n for n in t._nbrs.get(a, [])}
        old = links[a].get(b)
        links[a][b] = Neighbour(b, listed or (old.is_listed if old else False), site_of(a) == site_of(b))

    # same-site links (both directions), with existing and other new cells on the site
    for cid, s in zip(df.Cell_ID, df.Site_ID):
        for other in t._site_cells.get(s, set()) - {cid}:
            link(cid, other, False)
            link(other, cid, False)
        t._site_cells.setdefault(s, set()).add(cid)

    # listed links from each new cell's Neighbor_Cell_ID
    unknown, n_listed, n_reverse = set(), 0, 0
    for cid, raw in zip(df.Cell_ID, df.Neighbor_Cell_ID):
        for n in raw.replace(";", ",").split(","):
            n = n.strip()
            if not n or n == cid:
                continue
            if not exists(n):
                unknown.add(n)
                continue
            link(cid, n, True); n_listed += 1
            if add_reverse:
                link(n, cid, True); n_reverse += 1
    for a, d in links.items():
        t._nbrs[a] = list(d.values())

    new_rows = pd.DataFrame({"site_id": df.Site_ID.to_numpy(),
                             "technology": df.get("Technology", pd.Series([None] * len(df))).to_numpy(),
                             "region": df.get("Region", pd.Series([None] * len(df))).to_numpy(),
                             "district": df.get("District", pd.Series([None] * len(df))).to_numpy(),
                             "unseen_site": True}, index=pd.Index(df.Cell_ID.to_numpy(), name="cell_id"))
    t.cells = pd.concat([topo.cells, new_rows])
    report = {"new_cells": sorted(new_ids), "listed_links": n_listed, "reverse_links": n_reverse,
              "new_sites": sorted({s for s in df.Site_ID if s not in topo._site_cells}),
              "unknown_neighbour_ids": sorted(unknown)[:50], "n_unknown_neighbour_ids": len(unknown),
              "neighbours_per_new_cell": {c: len(t._nbrs.get(c, [])) for c in sorted(new_ids)}}
    return t, report
