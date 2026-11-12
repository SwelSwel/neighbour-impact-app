# Neighbour Impact Prediction: web app and API

When a mobile cell goes down, its users move to the neighbouring cells. This app predicts each neighbour's
**latency, utilization and packet loss** at the next 3-hourly reading, with an 80% range and a warning for
neighbours expected to degrade. Every input comes from before the outage.

The user only gives **a cell and an outage time**. The app looks up the other inputs (neighbours, their last three
readings, last week's reading, the failed cell's load) from a data store exported by the notebook. It then builds the
same 66 features and runs the three Gradient Boosting models trained in `Neighbour_Impact_Prediction_final.ipynb`.

| Page | The user gives | The user gets |
|---|---|---|
| 1 · Predict an outage | A real outage from the list, or any cell and time | Predictions for every neighbour next to the actual values; any input can be edited and re-predicted |
| 2 · What-if | An outage, then sliders: failed cell's load, cells already down, hour, neighbours also down | How the impact changes, and a load sweep (dose-response) |
| 3 · Upload data | A CSV or ZIP in Dataset 2 format | The same predictions for the user's own readings |
| 4 · API | JSON | Endpoint list, curl/Python examples, a request tester |

---

## Live application

| | Address |
|---|---|
| Website | `https://neighbour-ui.politefield-0864fbc1.eastasia.azurecontainerapps.io` *(from the latest GitHub Actions run summary)* |
| API docs | `https://neighbour-api.politefield-0864fbc1.eastasia.azurecontainerapps.io/docs` |

The website needs no login. The API's prediction endpoints need an `X-API-Key` header; the key is shared with the
evaluators separately and is never stored in this repository. `/health` and `/ready` need no key. Both apps scale to
zero when idle, so the first visit can take up to 30 seconds.

---

## 1. Get the files from the notebook

**Reproducing the notebook from the raw data.** The raw datasets are not in this repository (they are course data
and too large for GitHub). Put `Dataset_1_Cell_Information.csv` and `2025_01.csv` to `2025_06.csv` in one folder, and
point the notebook to it with the `TELECOM_DATA_DIR` environment variable (or edit the default path in section 1):

```bash
pip install -r requirements-notebook.txt
setx TELECOM_DATA_DIR "C:\path\to\Data"      # Windows (new terminal afterwards). macOS/Linux: export TELECOM_DATA_DIR=...
```

Running the notebook top to bottom reproduces every experiment (`RANDOM_STATE = 42`) and writes its outputs to
`<data folder>/impact_outputs/`, outside this repository. The app's `data/` and `models/` files then come from that
run, as follows.

1. Open `Neighbour_Impact_Prediction_final.ipynb` and `pip install pyarrow`. Then either:
   - **Kernel still has everything loaded** (you just ran the whole notebook): add a cell after section 19b, open
     `notebook/19c_export_app_data.py`, copy **its contents** into the cell and run it.
   - **Fresh kernel (normal case):** run only the cells of sections 1, 2, 3 and 5 (the grid loads from
     `kpi_grid_cache.npz`), then paste the contents of `notebook/19c_export_app_data_standalone.py` into a new cell
     and run it. It reloads `pairs.pkl` and the model bundle from your earlier run, so sections 6–19 are not re-run.

   Paste the code, not the file name: a cell containing only `notebook/...py` is read as a Jupyter command.
2. It writes `impact_outputs/app_data/`. Copy **everything in that folder** into this repository's `data/` folder.
3. Copy `neighbour_impact_models.joblib` and `deploy_test_cases.json` from `impact_outputs/` into `models/`
   (written by section 19b; skip this if they are already there from the same notebook run).

The data store and the model must come from **the same notebook run**: the tests compare the app's predictions with
the notebook's predictions for real test outages (`data/lookup_test_cases.json`).

### Data store mode (`STORE_MODE` at the top of the cell)

| Mode | What is stored | Size (June) | Page 1 can look up |
|---|---|---|---|
| `"outages"` (default) | Only the readings around real test-period outages | A few MB, fits in GitHub | Any real June outage |
| `"full"` | Every cell, every 3-hour slot, 24 May to 30 June | About 100 MB | Any cell at any time in that period |

With `"full"`, use Git LFS (or mount the folder, see docker-compose.yml) and give the API container 2 GiB of memory.
With `"outages"`, the data store is small. You can still edit any value on page 1, use the what-if page, upload new
data, or call `/predict` with any values.

---

## 2. Run it on your computer (no Docker)

Python 3.13, from the repository folder:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt -r requirements-ui.txt

pytest                             # all tests should pass

# terminal 1: the API on http://localhost:8000  (interactive docs: http://localhost:8000/docs)
$env:API_KEY="dev-key"             # PowerShell.  cmd: set API_KEY=dev-key   macOS/Linux: export API_KEY=dev-key
uvicorn app.main:app --port 8000 --reload

# terminal 2: the website on http://localhost:8501
cd ui
$env:API_KEY="dev-key"
$env:API_URL="http://localhost:8000"
streamlit run Home.py
```

## 3. Run it with Docker (the same containers that go to Azure)

```bash
cp .env.example .env               # then set API_KEY
docker compose up --build
# Website http://localhost:8501   ·   API http://localhost:8000/docs
```

## 4. Deploy to Azure

**Architecture.** Two containers on Azure Container Apps (region East Asia), images in Azure Container Registry,
container logs in Log Analytics. The website calls the API over HTTPS; only the API holds the model.

**One-time set-up** (Azure Cloud Shell, Bash):

1. Create the GitHub repository first (empty), then edit `GITHUB_REPO` (and `LOCATION` if needed) at the top of
   `infra/setup_azure.sh`, and run `bash infra/setup_azure.sh`. It creates the resource group, registry, log
   workspace, Container Apps environment, a managed identity that pulls images (no registry password), and the
   GitHub login (OpenID Connect: no Azure password or client secret is stored anywhere).
2. In GitHub → Settings: create the environment `production`; add the 4 secrets and 6 variables the script prints.
3. Push to `main`.

**CI/CD** (`.github/workflows/ci-cd.yml`):

| Event | What runs |
|---|---|
| Pull request into `main` | Tests only |
| Push to `main`, or "Run workflow" | Tests → build both images (tagged with the commit ID) → push to the registry → deploy the API → smoke test `/health`, `/ready`, `/predict`, `/predict-by-cell` and a bad request (422) → **roll back the API if the smoke test fails** → deploy the website and check its health |

Settings in the `az containerapp create` commands (CPU, memory, environment variables) apply only the first time an
app is created. To change them later, use `az containerapp update ... --set-env-vars` or `--cpu/--memory`.
For a `"full"` data store, the API needs `--memory 2.0Gi` and `--cpu 1.0`.

**Troubleshooting.**

| Error | Fix |
|---|---|
| `RequestDisallowedByAzure` when creating resources | The region is not allowed for the subscription; pick one from the list in the `LOCATION` comment |
| Login step: `AADSTS700213 No matching federated identity record` | Add a federated credential whose subject is exactly the one in the error message (`az ad app federated-credential create`) |

**Monitoring.** `/health` (process up) and `/ready` (model and data loaded) are used by Docker and Azure health
probes. The API writes one JSON log line per request, with a request ID that is also returned in the `x-request-id`
header. Logs are queried in the Log Analytics workspace, for example:

```
ContainerAppConsoleLogs_CL | where ContainerAppName_s == "neighbour-api" | order by TimeGenerated desc | take 50
```

**Cost.** Both apps scale to zero when idle (`--min-replicas 0`), so idle time costs almost nothing; the Basic
registry (about USD 5 a month) is the main fixed cost. Builds run on GitHub's free runners. To remove everything
after the course: `az group delete -n rg-neighbour-impact --yes`.

---

## 5. Where things are (edit what you need)

```
app/                         the API (FastAPI)
├── config.py                every setting, read from environment variables
├── main.py                  start-up (loads model + data once), logging, error handling, routers
├── deps.py                  shared state, API-key check
├── schemas.py               request/response shapes and input validation
├── features.py              the 66 features: same code as the notebook (do not change without retraining)
├── predictor.py             loads the model bundle, predicts, 80% intervals, prediction slot
├── service.py               request -> features -> predictions -> warnings (shared by all endpoints)
├── routers/
│   ├── system.py            /health /ready /model-info /data-info
│   ├── predict.py           /predict (any values)  /example
│   ├── lookup.py            /cells /outages /lookup /predict-by-cell (data store)
│   └── upload.py            /upload/outages /upload/lookup /upload/predict-by-cell (Dataset 2 file)
└── store/
    ├── grid.py              the 3-hourly time grid
    ├── topology.py          cells, sites, neighbour links (Dataset 1)
    ├── kpi_store.py         KPI readings indexed by (cell, slot)
    ├── outages.py           OUTAGE/RECOVERY pairing, "down" rule, cells already down on a site
    ├── dataset2.py          reads an uploaded Dataset 2 file with the notebook's rules
    ├── new_cells.py         reads an uploaded new-cells file (Dataset 1 format) for one request
    └── lookup.py            cell + time -> the full model input (mirrors notebook section 8)
ui/                          the website (Streamlit)
├── Home.py                  start page
├── pages/                   one file per page; the number in the name sets the order
└── lib/
    ├── settings.py          API address, labels, units, warning colours
    ├── api.py               every call to the API
    ├── pickers.py           outage list with filters; cell + time picker
    ├── case.py              the editable input tables
    ├── results.py           summary numbers and the results table
    ├── layout.py            page frame, inputs -> predict -> results section
    └── templates.py         column guides and templates for the upload page
notebook/                    the training notebook and the cells that export data/ and models/
infra/setup_azure.sh         one-time Azure set-up
.github/workflows/ci-cd.yml  the CI/CD pipeline
tests/                       consistency with the notebook, API behaviour, failure scenarios
```

| To change… | Edit |
|---|---|
| Warning thresholds (+20% degraded, +50% severe) | `DEGRADED_THRESHOLD`, `SEVERE_THRESHOLD` env vars. No rebuild needed |
| Allow any time with an `"outages"` store | `ALLOW_ANY_TIME=true`. Lookups between outages then return "no reading" errors |
| Max neighbours per request, upload size | `MAX_NEIGHBOURS`, `MAX_UPLOAD_MB` |
| Column names, units, colours on the website | `ui/lib/settings.py` |
| The results table (columns, highlighting) | `ui/lib/results.py` |
| The outage list filters | `ui/lib/pickers.py` |
| The what-if sliders | `ui/pages/2_What_if.py`, function `scenario()` |
| Add a page | New file `ui/pages/5_Name.py`; start it with `page("Title", "caption")` from `lib.layout` |
| Add an endpoint | New function in a file in `app/routers/` (or a new router added in `app/main.py`) |
| How neighbours or readings are looked up | `app/store/lookup.py`. Then run `pytest`: the notebook-consistency tests must still pass |

## 6. Tests

`pytest` runs 45 tests. The data-store tests are skipped if `data/` is empty.

- **Training/production consistency:**
  - Features and predictions match the notebook on 25 exported rows.
  - **Looking up real test outages by cell and time** gives the notebook's prediction and actual value for every
    neighbour.
  - **A Dataset 2 file** built from raw rows gives the same predictions.
- **Rules copied from the notebook:** time slots, bare-date timestamps, outage/recovery pairing, the "down" rule, and
  cells already down on a site.
- **Failure scenarios:**
  - missing or wrong API key (401);
  - invalid values, unknown fields, duplicate neighbours, too many neighbours (422);
  - an unknown cell (404);
  - a time outside the stored data, or not a real outage in `"outages"` mode (422);
  - a CSV with the wrong columns, no usable rows, or a corrupt ZIP (422);
  - a new-cells file with cells that already exist in Dataset 1 (422).
- **Missing values:** blank KPIs are accepted and treated as missing, as in training.

## 7. Limitations

- Lookups only cover the stored period (June 2025 plus one week before). A live system would read the same inputs
  from the operator's KPI database. Swapping that in means replacing `KPIStore`; the rest of the app stays the same.
- Uploaded files use the neighbour links from Dataset 1. Readings for cells that are not in Dataset 1 are ignored
  unless those cells are described in a new-cells file uploaded with them (page 3, option 2); new cells are not saved.
- What-if changes outside the range seen in training (for example 3× load) are extrapolations.
- The website has no login. It is a demo; a real deployment would add sign-in in front of it.
- The models were trained and evaluated on one operator's data from January to June 2025. Their accuracy on later
  data is not monitored automatically yet: a live system would compare each prediction with the reading that
  arrives three hours later and track the error over time.
