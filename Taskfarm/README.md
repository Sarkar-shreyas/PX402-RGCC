# Taskfarm/ — HPC Orchestration

This directory contains all Slurm job scripts and production YAML configs for running the IQHE and QSHE RG Monte Carlo pipeline at scale on the `<host>` HPC cluster. Nothing in this directory runs locally — it is staged onto the cluster via `file_management.py` and executed there by `sbatch`. The authoritative Python engine that these scripts invoke lives in `source/`.

File transfers between the local repository and the cluster are managed exclusively by [`file_management.py`](../file_management.py) at the repository root. See the [HPC Workflow section of the root README](../README.md#hpc-workflow) for the full push → submit → pull sequence.

---

## Environment requirements

- SSH access to `<host>` configured under the local `~/.ssh/config` (the `HOST` `.env` variable must resolve without a password prompt)
- `.env` variables consumed by `file_management.py` at transfer time:
  - `HOST` — SSH hostname
  - `USERNAME` — remote username
  - `REMOTE_DIR` — remote base directory
- On the cluster, the following must be available before submitting any job:
  - `GCC/13.3.0` and `SciPy-bundle/2024.05` modules (loaded by each script via `module load`)
  - A project virtualenv at `$REMOTE_DIR/.venv` providing PyYAML (activated by each script)
  - The pushed `source/` tree present at `$REMOTE_DIR/code/source/`

---

## scripts/

Scripts fall into two categories: **entry-point wrappers** (called directly with `bash`) and **Slurm job scripts** (submitted internally via `sbatch`; not invoked by hand).

### Entry-point wrappers

| Script | Purpose | Usage |
|--------|---------|-------|
| [run_rg.sh](scripts/run_rg.sh) | Submit a Fixed-Point (FP) IQHE RG run — validates/updates the YAML config via `parse_config.py`, then hands off to `rg_fp_master.sh` | `bash Taskfarm/scripts/run_rg.sh --config Taskfarm/configs/iqhe.yaml [--set "key=val"] [--out /tmp/configs]` |
| [run_shifts.sh](scripts/run_shifts.sh) | Submit an EXP (perturbed) shifted-RG run — selects one shift from the config's `data_settings.shifts` list by index, then submits `shifted_rg.sh` | `bash Taskfarm/scripts/run_shifts.sh --config Taskfarm/configs/iqhe.yaml --index 0 [--set "key=val"] [--out /tmp/configs]` |
| [run_qp.sh](scripts/run_qp.sh) | Submit a QSHE (q, p) parameter-sweep job — partitions the q-axis into blocks for parallel Slurm array execution, then submits `qp_trials.sh` | `bash Taskfarm/scripts/run_qp.sh --config Taskfarm/configs/qshe.yaml [--q-block-size 10] [--vars p,q]` |
| [run_gen.sh](scripts/run_gen.sh) | Submit a single QSHE q-block generation job — low-level wrapper around `temp_gen.sh`, used when re-running a specific block | Internal / recovery use |

### Slurm job scripts (submitted internally)

These scripts contain the `#SBATCH` resource directives and are submitted by the wrappers above via `sbatch`. They are not intended to be called directly.

| Script | Slurm job name | Role |
|--------|---------------|------|
| [rg_fp_master.sh](scripts/rg_fp_master.sh) | `rg_fp_master` | Orchestrates the full FP RG iteration loop — chains `rg_gen_batch.sh` and `rg_hist_manager.sh` as dependent array jobs for each RG step |
| [rg_gen_batch.sh](scripts/rg_gen_batch.sh) | `rg_gen` | Slurm array job (32 tasks, 4 concurrent) — generates one batch of `t'` samples for a single RG step via `source.data_generation` |
| [rg_hist_manager.sh](scripts/rg_hist_manager.sh) | `rg_hist` | Aggregates per-batch sample files into a single histogram NPZ for one RG step via `source.histogram_manager` and `source.helpers` |
| [shifted_rg.sh](scripts/shifted_rg.sh) | `shifted_rg_master` | Orchestrates an EXP (shifted) RG iteration loop — mirrors `rg_fp_master.sh` but starts from the shifted initial distribution produced by `source.shift_z` |
| [gen_shifted_data.sh](scripts/gen_shifted_data.sh) | `gen_shift` | Slurm array job — generates shifted initial `t` samples for one EXP run step via `source.shift_z` |
| [qp_trials.sh](scripts/qp_trials.sh) | `qp_trials` | Runs QSHE MC trials for one q-block via `source.qshe_data_gen`; one array element per block |
| [temp_gen.sh](scripts/temp_gen.sh) | `temp_gen` | Single-node QSHE q-block generation job (non-array variant of `qp_trials.sh`; used for reruns) |
| [agg.sh](scripts/agg.sh) / [temp_agg.sh](scripts/temp_agg.sh) | `temp_agg` | Aggregates per-q-block QSHE output arrays into a single file via `source.qshe_data_agg` |

---

## configs/

### [iqhe.yaml](configs/iqhe.yaml) — IQHE production config

Targets the `<host>` Slurm cluster for IQHE FP and EXP runs. Submit via `run_rg.sh` (FP) or `run_shifts.sh` (EXP).

| Key | HPC value | Local value | Reason for difference |
|-----|-----------|-------------|----------------------|
| `rg_settings.samples` | `480000000` (480M) | `32000000` (32M) | 10× larger sample count for production accuracy |
| `rg_settings.steps` | `9` | `7` | More steps needed to reach convergence at higher N |
| `rg_settings.seed` | `12345` | `1234` | Different seeds keep local and HPC runs statistically independent |
| `main.id` | `"hpc"` | `"local"` | Environment tag used in log output |
| `main.output_folder` | `""` (resolves to `REMOTE_DIR`) | `"Local data"` | Output root is the cluster's remote dir, not a local path |

All other keys (`engine`, `parameter_settings`, `convergence`, `data_settings`) are identical between HPC and local IQHE configs.

### [qshe.yaml](configs/qshe.yaml) — QSHE production config

Targets the `<host>` Slurm cluster for QSHE q-p parameter-sweep runs. Submit via `run_qp.sh`.

| Key | Value | Notes |
|-----|-------|-------|
| `engine.model` | `"qshe"` | Selects the QSHE matrix solver |
| `engine.method` | `"numerical"` | QSHE requires the 20×20 matrix solver; analytic path is not available |
| `engine.expr` | `"shreyas"` | QSHE-specific RG map parametrisation |
| `rg_settings.steps` | `15` | More RG iterations than IQHE (convergence is slower for the 2D q-p sweep) |
| `rg_settings.metric` | `"all"` | Returns all per-trial observable values (not just mean/median) |
| `parameter_settings.q` | `[0.0, 0.5]`, 51 points | q-axis sweep range and resolution |
| `parameter_settings.p` | `[0.001, 0.999]`, 500 points | p-axis sweep range and resolution |
