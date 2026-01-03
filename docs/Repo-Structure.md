Local repo layout vs Remote staged layout

- Local (git):
  - `source/` — shared Python library used by local and cluster runs
  - `Taskfarm/scripts/` — local copies of runtime shell scripts
  - `Taskfarm/configs/` — YAML templates and configs
  - `Local/` — local helper and test drivers (this folder is local-only)

- Remote (staged on `<HOST>` via `file_management.py`):
  - `<REMOTE_ROOT>/code/source/` — Python runtime library (pushed from local `source/`)
  - `<REMOTE_ROOT>/scripts/` — shell scripts & YAML configs (pushed from `Taskfarm/scripts/` and `Taskfarm/configs/`)
  - `<REMOTE_ROOT>/job_outputs/` — job outputs, organized by version and FP/EXP

Important: `Local/` is not part of the remote staged layout; `Local/run_local.py` is intended for local testing only.

# Repo structure and mental model

Top-level folders (observed):

- `Taskfarm/` — HPC orchestration scripts and Taskfarm configs (authoritative pipeline entrypoints). See `Taskfarm/scripts/*.sh` and `Taskfarm/configs/`.
- `Local/` — local drivers and config templates for quick testing. Key file: [Local/run_local.py](../Local/run_local.py) and `Local/configs/local_iqhe.yaml`.
- `source/` — core Python modules used by both cluster and local code. Key modules: `source/config.py`, `source/parse_config.py`, `source/utilities.py`, `source/data_generation.py`, `source/histogram_manager.py`.
- `analysis/` — plotting and postprocessing utilities for analysis and reporting (not part of core pipeline orchestration).
- `Data from taskfarm/` — archived / produced datasets (per-version subfolders).

Runtime mental model

- Runtime on cluster (authoritative): Taskfarm scripts submit and coordinate many per-task jobs that produce partial histograms. Those per-task outputs are then aggregated and used by the RG driver.
- Local-only: `Local/run_local.py` implements the driver loop (`rg_fp()`, `rg_exp()`) and is intended for small-scale testing and debugging. It mirrors the cluster flow but runs everything inside one process.

Import / module conventions

- `source/` contains shared library code imported by `Local/run_local.py` and by any Taskfarm worker code. Examples:
  - `from source.parse_config import build_parser, validate_input` in `Local/run_local.py`.
  - `from source.utilities import rg_data_workflow, launder, save_data` in `Local/run_local.py`.
- `source` is intentionally not an installable package via pip for the following reasons;
  - Cluster portability
  - Avoid editable installs on HPCs
  - Consistent import paths for slurm jobs and local testing

Where to start when contributing

- If you need to change runtime behavior for the pipeline, modify `source/utilities.py` and `source/config.py` and update the Taskfarm scripts in `Taskfarm/scripts/`.
- For small-scale testing, use `Local/run_local.py` and `Local/configs/local_iqhe.yaml`.
