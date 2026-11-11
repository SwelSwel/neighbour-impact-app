"""Cells, sites and neighbour links (Dataset 1). Which cells can take over a failed cell's users."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Neighbour:
    cell_id: str
    is_listed: bool
    is_same_site: bool


@dataclass
class Topology:
    cells: pd.DataFrame                    # index cell_id; columns site_id, technology, region, district, unseen_site
    links: pd.DataFrame                    # cell_id, nbr_id, is_listed, is_same_site
    _nbrs: dict[str, list[Neighbour]] = field(default_factory=dict, repr=False)
    _site_cells: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.cells.index.name != "cell_id":
            self.cells = self.cells.set_index("cell_id")
        for cid, g in self.links.groupby("cell_id", sort=False):
            self._nbrs[cid] = [Neighbour(str(n), bool(l), bool(s))
                               for n, l, s in zip(g.nbr_id, g.is_listed, g.is_same_site)]
        for site, g in self.cells.groupby("site_id", sort=False):
            self._site_cells[str(site)] = set(g.index.astype(str))

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dir(cls, path: Path) -> "Topology":
        """Files written by notebook section 19c."""
        return cls(pd.read_parquet(path / "cells.parquet"), pd.read_parquet(path / "neighbours.parquet"))

    @classmethod
    def from_master_csv(cls, path: Path, source: str = "listed+site") -> "Topology":
        """Build the same links directly from Dataset 1, using the notebook's rules (section 2)."""
        m = pd.read_csv(path, low_memory=False)
        for c in ["Site_ID", "Cell_ID"]:
            m[c] = m[c].astype(str).str.strip()
        m = m.drop_duplicates("Cell_ID").reset_index(drop=True)
        known = set(m.Cell_ID)
        site = dict(zip(m.Cell_ID, m.Site_ID))
        listed = set()
        for cid, raw in zip(m.Cell_ID, m.Neighbor_Cell_ID.fillna("").astype(str)):
            for n in raw.replace(";", ",").split(","):
                n = n.strip()
                if n and n != cid and n in known:
                    listed.add((cid, n))
        same = {(a, b) for _, g in m.groupby("Site_ID") for a in g.Cell_ID for b in g.Cell_ID if a != b}
        pairs = {"listed": listed, "site": same, "listed+site": listed | same}[source]
        links = pd.DataFrame(sorted(pairs), columns=["cell_id", "nbr_id"])
        links["is_listed"] = [p in listed for p in zip(links.cell_id, links.nbr_id)]
        links["is_same_site"] = [site[a] == site[b] for a, b in zip(links.cell_id, links.nbr_id)]
        cells = pd.DataFrame({"cell_id": m.Cell_ID, "site_id": m.Site_ID, "technology": m.get("Technology"),
                              "region": m.get("Region"), "district": m.get("District"), "unseen_site": False})
        return cls(cells, links)

    # ---------------------------------------------------------------- queries
    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells.index

    def site_of(self, cell_id: str) -> str:
        return str(self.cells.at[cell_id, "site_id"])

    def cells_on_site(self, site_id: str) -> set[str]:
        return self._site_cells.get(site_id, set())

    def neighbours(self, cell_id: str) -> list[Neighbour]:
        return self._nbrs.get(cell_id, [])

    def info(self, cell_id: str) -> dict:
        r = self.cells.loc[cell_id]
        return {"cell_id": cell_id, "site_id": str(r.site_id), "technology": r.get("technology"),
                "region": r.get("region"), "district": r.get("district"),
                "n_neighbours": len(self.neighbours(cell_id))}

    def search(self, q: str = "", limit: int = 50) -> list[str]:
        ids = self.cells.index
        if q:
            ids = ids[ids.str.contains(q, case=False, regex=False)]
        return [str(i) for i in ids[:limit]]
