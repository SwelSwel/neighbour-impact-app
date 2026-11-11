Copy everything from your notebook's `impact_outputs/app_data/` folder here (written by section 19c):

- `cells.parquet`, `neighbours.parquet`, `outages.parquet`, `kpi.parquet`, `meta.json`, `lookup_test_cases.json`

The API loads this folder at start-up (`DATA_DIR`, default `data`). Without it, `/predict` and the API page still
work, but pages 1–3 cannot look anything up.
