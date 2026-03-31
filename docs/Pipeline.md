# Pipeline

This document describes the end-to-end runtime lifecycle for both the IQHE and QSHE
pipelines.  Read [Config.md](Config.md) alongside this document for a full description
of every configuration key.

---

## Overview

The pipeline extracts the critical exponent ν at quantum phase transitions using Monte
Carlo Renormalisation Group (RG) methods.  There are two independent pipelines sharing
the same `source/` library:

| Pipeline | Model | Analysis method | Driver |
|----------|-------|-----------------|--------|
| IQHE | Integer Quantum Hall Effect | RSRG perturbation (FP + EXP runs) | `Local/run_local_iqhe.py` / Slurm |
| QSHE | Quantum Spin Hall Effect | FSS of RG flow over (p, q) grid | `test_qshe.ipynb` |

Both pipelines share `source/utilities.py`, `source/config.py`, and the config system.
The QSHE pipeline has no CLI analysis driver — all post-processing is done in the
notebook.  See [QSHE.md](QSHE.md) for full details.

---

## Deployment model

| Location | Contents |
|----------|----------|
| Local git repo | `source/`, `Taskfarm/scripts/`, `Taskfarm/configs/`, `Local/` |
| Remote `<REMOTE_ROOT>/code/source/` | Staged Python library (pushed from `source/`) |
| Remote `<REMOTE_ROOT>/scripts/` | Staged shell scripts and YAML configs |
| Remote `<REMOTE_ROOT>/job_outputs/` | Job outputs, organised by version and run type |

`file_management.py` is the authoritative tool for all push and pull operations; see
[Runbook.md](Runbook.md) for usage.

---

## IQHE pipeline

### Fixed-Point (FP) run

The FP run iterates the RG map from a flat initial distribution until the z-histogram
converges to the critical fixed point.

```
source/data_generation.py
  → generates t' samples for one RG step
  → writes t_data_RG{step}_{size}_samples.npy

source/helpers.py  (launder / symmetrise)
  → converts t → z, optionally symmetrises the z-distribution
  → writes z arrays and laundered t batches

source/histogram_manager.py
  → builds / appends compressed NPZ histograms

Repeat for rg_settings.steps iterations.
```

**Output layout:**

```
{version}/FP/
├── hist/t/t_hist_RG{i}.npz
├── hist/z/z_hist_RG{i}.npz
├── hist/sym_z/sym_z_hist_RG{i}.npz   (when engine.symmetrise = 1)
├── stats/{statistic}_moments.json
└── config/updated_config.yaml
```

### EXP (shifted) run

The EXP run loads the final FP distribution, applies a small perturbation shift δ to
the z-distribution (`source/shift_z.py`), and then runs the same RG iteration.  The
growth rate of the peak displacement with δ yields the critical exponent ν.

**Output layout:**

```
{version}/EXP/shift{s}/hist/
├── t/t_hist_RG{i}.npz
└── z/z_hist_unsym_RG{i}.npz
```

### ν extraction

```bash
python -m analysis.critical_exponent \
    --version VERSION --mode EXP --steps N [--loc local|taskfarm]
```

`analysis/critical_exponent.py` reads the FP and EXP NPZ files, estimates the
z-distribution peak for each (shift, step) pair via `estimate_z_peak` (bootstrapped
Gaussian fitting on the top-5% bins), fits the log(peak displacement) vs log(shift)
relation with `fit_z_peaks`, and converts the slope to ν via `calculate_nu` from
`source/utilities.py`.

### Cluster execution (IQHE)

The Slurm orchestration is driven by scripts in `Taskfarm/scripts/`:

```
run_rg.sh
  → validates config, writes updated_config.yaml
  → sbatch rg_fp_master.sh <updated_config>

rg_fp_master.sh
  └─ for each RG step:
       sbatch rg_gen_batch.sh   (array job: python -m source.data_generation)
       sbatch --dependency=afterok rg_hist_manager.sh
                (python -m source.helpers + python -m source.histogram_manager)
```

For EXP runs:

```
run_shifts.sh --index N
  → sbatch shifted_rg.sh <updated_config>
```

`rg_gen_batch.sh` is submitted as a Slurm array (`--array=0-31%4`).  Each task writes
output to a local scratch directory under `$SLURM_TMPDIR`, then rsyncs to the shared
filesystem with a `READY` sentinel file.  `rg_hist_manager.sh` waits for all `READY`
markers before aggregating.

The generation batch script is invoked as:

```
python -m source.data_generation \
    ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED [EXISTING_T_FILE]
```

The histogram manager script is invoked as:

```
python -m source.helpers PROCESS ARRAY_SIZE INPUT OUTPUT SEED
python -m source.histogram_manager PROCESS VAR INPUT_FILE [EXISTING] OUTPUT_FILE RG_STEP [SHIFT]
```

See [Runbook.md](Runbook.md) for the full push → run → pull workflow.

---

## QSHE pipeline

### Data generation (HPC)

The QSHE pipeline scans a two-dimensional (q, p) grid.  For each grid cell it runs
`num_samples` Monte Carlo trials through `num_steps` RG iterations, recording the
per-step statistics of p and q.

```
source/qshe_data_gen.py  (one Slurm array element per q-block)
  → qp_trials() from source/utilities.py
  → writes p_data_q{b}_{N}_samples.npy  and  q_data_q{b}_{N}_samples.npy
  → writes DONE sentinel

source/qshe_data_agg.py  (single aggregation job)
  → checks DONE sentinels for every block
  → concatenates blocks along the q-axis
  → writes p_data_agg.npy, q_data_agg.npy
  → writes trial_state.json
```

**Output layout:**

```
{DATA_DIR}/{dataversion}/QP/
├── data/
│   ├── p_data_agg.npy              shape (q_num, p_num, steps, met_dim)
│   ├── q_data_agg.npy              shape (q_num, p_num, steps, met_dim)
│   └── q{b}/                       per-block working files (may be cleaned up)
│       ├── p_data_q{b}_{N}_samples.npy
│       ├── q_data_q{b}_{N}_samples.npy
│       └── DONE
├── config/updated_config.yaml
└── output/trial_state.json
```

### Analysis (local, Jupyter)

All QSHE analysis is performed in `test_qshe.ipynb`.  There is no CLI analysis
equivalent.  The notebook loads the aggregated NPY files and `updated_config.yaml`
from `{DATA_DIR}/{dataversion}/QP/`, post-processes RG flow trajectories, and extracts
ν via a finite-size scaling analysis.  Full details are in [QSHE.md](QSHE.md).

---

## Shared source/ library

Both pipelines import from `source/` without an installable package:

| Module | Role |
|--------|------|
| `source/utilities.py` | Core RG mathematics, sampling, transformations, histograms |
| `source/config.py` | Config loading → `IQHEConfig` / `QSHEConfig` dataclasses |
| `source/data_generation.py` | Batch sample generation for one IQHE RG step |
| `source/histogram_manager.py` | Build / append NPZ histograms |
| `source/helpers.py` | Laundering, symmetrisation, z ↔ t conversion |
| `source/fitters.py` | Peak estimation, linear fitting, ν calculation |
| `source/shift_z.py` | Apply perturbation shift to z-distribution (EXP runs) |
| `source/parse_config.py` | CLI argument parsing and config validation |
| `source/qshe_data_gen.py` | QSHE q-p trial generation (HPC) |
| `source/qshe_data_agg.py` | QSHE trial aggregation (HPC) |

`source/` is not installed as a package; it is imported directly via Python's module
search path.  On the cluster the library is staged to
`<REMOTE_ROOT>/code/source/` by `file_management.py`.
