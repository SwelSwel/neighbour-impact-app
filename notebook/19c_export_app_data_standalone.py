# ===================================================================================================================
# 19c (standalone). Data store for the web app, WITHOUT re-running the slow sections (6-19).
#
# Run these cells first (each takes seconds to a few minutes, the grid comes from kpi_grid_cache.npz):
#     section 1 (configuration), 2 (master table), 3 (read monthly files), 5 (outage events)
# Then run this cell. It loads what your earlier full run saved in impact_outputs/:
#     pairs.pkl                        (section 7)  -> rebuilds sections 8, 9 and 10 exactly
#     neighbour_impact_models.joblib   (section 19b) -> the deployed models' predictions for the test cases
# ===================================================================================================================
import joblib

# ---- needed globals normally defined in sections 6-7 -----------------------------------------------------------
WEEK = 7 * 24 // GRID_STEP_H
FAIL_VARS = ["Traffic", "Connected_Users", "Load", "Capacity", "Utilization", "Throughput"]
NORMAL_VARS = IMPACT_TARGETS + ["Traffic", "Load"]

def kpi_block(ci, j, prefix=""):
    """9 current values + lag-1, lag-2, mean of the two, trend2 (6-hour lookback) at slot j (as in section 6)."""
    d = {}
    for v in VARS9:
        g = G[v]; cur, l1, l2 = g[ci, j], g[ci, j - 1], g[ci, j - 2]
        d[f"{prefix}{v}"] = cur; d[f"{prefix}{v}_lag1"] = l1; d[f"{prefix}{v}_lag2"] = l2
        d[f"{prefix}{v}_mean2"] = (l1 + l2) / 2; d[f"{prefix}{v}_trend2"] = cur - l2
    return pd.DataFrame(d)

# ---- section 8: modelling pairs (from the saved pairs.pkl) -----------------------------------------------------
PAIRS = pd.read_pickle(OUTPUT_DIR / "pairs.pkl")
MD = PAIRS[PAIRS.in_outage].copy().reset_index(drop=True)
for v in IMPACT_TARGETS:
    MD[f"chg_{v}"] = MD[f"y_{v}"] - MD[f"before_{v}"]

# ---- section 9: splits A and B (same random seed, so the same unseen sites) ----------------------------------
rng = np.random.default_rng(RANDOM_STATE)
unseen_site = np.zeros(len(SITES), bool)
unseen_site[rng.choice(len(SITES), size=int(round(UNSEEN_SITE_FRACTION * len(SITES))), replace=False)] = True
fu = unseen_site[site_of_cell[MD.fail_ci.to_numpy()]]
nu = unseen_site[site_of_cell[MD.nbr_ci.to_numpy()]]
dev = ~fu & ~nu
MD["split_A"] = np.select([dev & (MD.start <= TRAIN_END), dev & (MD.start > TRAIN_END) & (MD.start <= VALID_END),
                           dev & (MD.start > VALID_END)], ["train", "valid", "test"], "-")
MD["unseen_june"] = fu & (MD.start > VALID_END)

# ---- section 10: features --------------------------------------------------------------------------------------
Xk = kpi_block(MD.nbr_ci.to_numpy(), MD.se.to_numpy() - 1, prefix="nb_"); Xk.index = MD.index
MD = pd.concat([MD.drop(columns=[c for c in Xk.columns if c in MD.columns]), Xk], axis=1)
MD["hour_sin"] = np.sin(2 * np.pi * MD.hour / 24); MD["hour_cos"] = np.cos(2 * np.pi * MD.hour / 24)
SETS = {"test": MD[MD.split_A == "test"], "unseen_june": MD[MD.unseen_june]}

# ---- deployed models (section 19b bundle) -> predictions for the test sets -------------------------------------
bundle = joblib.load(OUTPUT_DIR / "neighbour_impact_models.joblib")
FEATURES, DEPLOY_MODEL = bundle["features"], bundle.get("deploy_model", "Gradient Boosting")
PRED = {t: {DEPLOY_MODEL: {k: bundle["targets"][t]["model"].predict(S[FEATURES].to_numpy(np.float32))
                           for k, S in SETS.items()}} for t in IMPACT_TARGETS}
print(f"Rebuilt {len(MD):,} modelling pairs | test {len(SETS['test']):,} | unseen June {len(SETS['unseen_june']):,} "
      f"| model: {DEPLOY_MODEL}, {len(FEATURES)} features")

import json, datetime as _dt

STORE_MODE = "outages"                     # "outages" or "full"
N_LOOKUP_TEST_OUTAGES = 8                  # per split, written to lookup_test_cases.json
STORE_DIR = OUTPUT_DIR / "app_data"
STORE_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = VALID_END + pd.Timedelta(seconds=1)                      # first moment of the test period (June)
j_test0 = int(np.ceil((TEST_START - GRID_START) / STEP))
j0 = max(j_test0 - WEEK - LOOKBACK_STEPS - 1, 0)                     # lookback + last week need ~8 days before it
j1 = N_T - 1
win_start, win_end = GRID_TIMES[j0], GRID_TIMES[j1]
print(f"Stored period: {win_start} to {win_end} (slots {j0}-{j1}) | test period starts {TEST_START} | mode {STORE_MODE}")

# ---- 1. Cells (Dataset 1) -------------------------------------------------------------------------------------------
cells_df = pd.DataFrame({
    "cell_id": CELLS, "site_id": np.asarray(SITES)[site_of_cell].astype(str),
    "technology": master["Technology"].astype(str).to_numpy(), "region": master["Region"].astype(str).to_numpy(),
    "district": master["District"].astype(str).to_numpy(), "unseen_site": unseen_site[site_of_cell]})
cells_df.to_parquet(STORE_DIR / "cells.parquet", index=False)

# ---- 2. Neighbour links actually used by the model (NEIGHBOUR_SOURCE) ---------------------------------------------
coo = A_USE.tocoo()
listed_keys = A_NBR.tocoo(); listed_keys = listed_keys.row.astype(np.int64) * N_CELLS + listed_keys.col
use_keys = coo.row.astype(np.int64) * N_CELLS + coo.col
nbr_df = pd.DataFrame({"cell_id": CELLS[coo.row], "nbr_id": CELLS[coo.col],
                       "is_listed": np.isin(use_keys, listed_keys),
                       "is_same_site": site_of_cell[coo.row] == site_of_cell[coo.col]})
nbr_df.to_parquet(STORE_DIR / "neighbours.parquet", index=False)

# ---- 3. Outages overlapping the stored period ------------------------------------------------------------------------
ov = out[(out.start <= win_end) & (out.up.isna() | (out.up >= win_start))].copy()
split_of = {}
for k, label in [("unseen_june", "unseen site"), ("test", "test")]:
    for e in SETS[k].event_id.unique(): split_of[e] = label
n_pairs = pd.concat([SETS["test"], SETS["unseen_june"]]).groupby("event_id").size()
out_df = pd.DataFrame({
    "event_id": ov.event_id.to_numpy(), "cell_id": CELLS[ov.ci.to_numpy()], "start": ov.start.to_numpy(),
    "up": ov.up.to_numpy(), "duration_h": ov.duration_h.to_numpy(),
    "in_test_period": (ov.start > VALID_END).to_numpy(),
    "split": [split_of.get(e, "") for e in ov.event_id], "n_model_rows": [int(n_pairs.get(e, 0)) for e in ov.event_id]})
out_df.to_parquet(STORE_DIR / "outages.parquet", index=False)

# ---- 4. KPI readings ------------------------------------------------------------------------------------------------
if STORE_MODE == "full":
    ci, jj = np.nonzero(has_kpi[:, j0:j1 + 1]); jj = jj + j0
else:
    sel = np.nonzero(((out.start > VALID_END) & (out.start <= win_end)).to_numpy())[0]
    parts_c, parts_j = [], []
    for i in sel:
        c, se = int(out.ci.iat[i]), int(s_slot[i])
        cells_ = np.append(A_USE.indices[A_USE.indptr[c]:A_USE.indptr[c + 1]], c)
        for j in (se - LOOKBACK_STEPS - 1, se - LOOKBACK_STEPS, se - 1, se, se - WEEK):
            if 0 <= j < N_T:
                parts_c.append(cells_); parts_j.append(np.full(len(cells_), j))
    ci, jj = np.concatenate(parts_c), np.concatenate(parts_j)
    keep = has_kpi[ci, jj]; ci, jj = ci[keep], jj[keep]
    key = np.unique(ci.astype(np.int64) * N_T + jj); ci, jj = (key // N_T).astype(np.int64), (key % N_T).astype(np.int64)

kpi_df = pd.DataFrame({"cell_id": pd.Categorical(CELLS[ci]), "slot": jj.astype(np.int32),
                       "timestamp": GRID_TIMES[jj]})
for v in RAW8:
    kpi_df[v] = G[v][ci, jj].astype(np.float32)
kpi_df["down"] = down[ci, jj]
kpi_df.to_parquet(STORE_DIR / "kpi.parquet", index=False, compression="zstd")

# ---- 5. Meta ----------------------------------------------------------------------------------------------------------
meta = {"mode": STORE_MODE, "grid_start": str(GRID_START), "grid_step_h": GRID_STEP_H, "week_slots": int(WEEK),
        "lookback_steps": LOOKBACK_STEPS, "period_start": str(win_start), "period_end": str(win_end),
        "test_period_start": str(TEST_START), "neighbour_source": NEIGHBOUR_SOURCE, "raw_kpis": RAW8,
        "n_cells": int(len(cells_df)), "n_links": int(len(nbr_df)), "n_outages": int(len(out_df)),
        "n_kpi_rows": int(len(kpi_df)), "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
(STORE_DIR / "meta.json").write_text(json.dumps(meta, indent=1))

# ---- 6. Lookup test cases: real outages with this notebook's predictions for each neighbour ---------------------------
lookup_cases = []
for k in ["test", "unseen_june"]:
    S = SETS[k]
    if len(S) == 0: continue
    preds = {t: PRED[t][DEPLOY_MODEL][k] for t in IMPACT_TARGETS}
    evs = pd.Series(S.event_id.unique()).sample(n=min(N_LOOKUP_TEST_OUTAGES, S.event_id.nunique()), random_state=RANDOM_STATE)
    for e in evs:
        m = (S.event_id == e).to_numpy(); rows = S[m]
        nb = {}
        for pos, (_, r) in enumerate(rows.iterrows()):
            nb[str(CELLS[int(r.nbr_ci)])] = {
                "predicted": {t: float(preds[t][m][pos]) for t in IMPACT_TARGETS},
                "actual": {t: float(r[f"y_{t}"]) for t in IMPACT_TARGETS}}
        lookup_cases.append({"split": k, "event_id": int(e), "cell_id": str(CELLS[int(rows.fail_ci.iat[0])]),
                             "outage_start": rows.start.iat[0].isoformat(), "n_absorbing": int(rows.n_absorbing.iat[0]),
                             "neighbours": nb})
(STORE_DIR / "lookup_test_cases.json").write_text(json.dumps(lookup_cases, indent=1))

for f in sorted(STORE_DIR.iterdir()):
    print(f"  {f.name:<24} {f.stat().st_size / 1e6:8.2f} MB")
print(f"KPI rows: {len(kpi_df):,} | outages stored: {len(out_df):,} "
      f"(test period: {int(out_df.in_test_period.sum()):,}, in model test sets: {int((out_df.split != '').sum()):,}) | "
      f"lookup test cases: {len(lookup_cases)}")
print("Copy the whole app_data folder into the app's data/ folder.")
