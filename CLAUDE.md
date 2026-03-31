# CLAUDE.md — RG Monte Carlo Pipeline

## Project Overview

This is a **Renormalization Group (RG) Monte Carlo pipeline** for studying phase transitions in quantum condensed matter systems, specifically:

- **IQHE** (Integer Quantum Hall Effect) — 2D electron systems under strong magnetic fields
- **QSHE** (Quantum Spin Hall Effect) — topological edge states with spin-orbit coupling

The pipeline runs locally for testing (32M samples) and on a Slurm HPC cluster (`vulcan2`) for production (320M samples). It extracts the **critical exponent ν** characterising the divergence of the correlation length at the quantum phase transition.

---

## Directory Structure

```
Project Code/
├── source/               # Core RG engine (authoritative)
├── Local/                # Local testing drivers & configs
├── Taskfarm/             # HPC orchestration scripts & configs
│   ├── scripts/          # Slurm shell scripts
│   └── configs/          # Production YAML configs
├── analysis/             # Post-processing, plotting, ν extraction
├── QSHE/                 # Experimental QSHE code
├── testing/              # Solver validation tests
├── docs/                 # Pipeline docs, runbooks, config reference
├── Data from taskfarm/   # HPC outputs (read-only, not in context)
├── Local data/           # Local run outputs (not in context)
├── report/               # Thesis report (not in context)
├── constants.py          # Global constants and paths (loads .env)
├── file_management.py    # scp/rsync transfer utility
├── requirements.txt      # numpy>=1.26, scipy>=1.11, matplotlib>=3.8, PyYAML>=6.0
└── .env                  # Local path config (not committed)

```

---

## Key Source Files

| File | Role |
|------|------|
| [source/utilities.py](source/utilities.py) | Core RG math: sampling, transformations, histograms |
| [source/config.py](source/config.py) | Config loading/parsing → `IQHEConfig`/`QSHEConfig` dataclasses |
| [source/data_generation.py](source/data_generation.py) | Batch sample generation for one RG step |
| [source/histogram_manager.py](source/histogram_manager.py) | Build/append NPZ histograms |
| [source/helpers.py](source/helpers.py) | Laundering, symmetrisation, z↔t conversion |
| [source/fitters.py](source/fitters.py) | Peak estimation, Gaussian fitting, ν calculation |
| [source/shift_z.py](source/shift_z.py) | Apply perturbation shift to z-distribution (EXP runs) |
| [source/parse_config.py](source/parse_config.py) | CLI argument parsing and config validation |
| [source/qshe_data_gen.py](source/qshe_data_gen.py) | QSHE q-p trial generation |
| [source/qshe_data_agg.py](source/qshe_data_agg.py) | QSHE trial aggregation |
| [Local/run_local_iqhe.py](Local/run_local_iqhe.py) | Local IQHE FP/EXP driver |
| [Local/run_local_qshe.py](Local/run_local_qshe.py) | Local QSHE driver |
| [analysis/critical_exponent.py](analysis/critical_exponent.py) | Extract ν from FP+EXP histograms |
| [constants.py](constants.py) | Global constants (N, SHIFTS, CURRENT_VERSION, plot params) |
| [file_management.py](file_management.py) | Push/pull files to/from HPC cluster |

---

## Configuration System

### Config Hierarchy

1. **YAML config file** (`Taskfarm/configs/iqhe.yaml` or `Local/configs/local_iqhe.yaml`)
2. **CLI overrides** via `--set "key.nested.path=value"` (parsed by `parse_config.py`)
3. **Validated dataclass** (`IQHEConfig` or `QSHEConfig`) via `build_config(dict)`

### Config Structure (YAML)

```yaml
main:
  version: "fp_iqhe_numerical_shaw"  # run identifier
  type: "fp"                          # "fp" or "exp"
engine:
  model: "iqhe"           # "iqhe" or "qshe"
  method: "analytic"      # "analytic" (4 phases) or "numerical" (8 phases, matrix)
  expr: "shaw"            # expression type
  symmetrise: 1           # 0=off, 1=symmetrise z-distribution
  resample: "i"           # resampling mode (inverse CDF)
rg_settings:
  seed: 1234
  steps: 7                # RG iterations (local: 7, HPC: 9)
  samples: 32000000       # total MC samples (local: 32M, HPC: 320M)
  matrix_batch_size: 100000
data_settings:
  inputs: [1.0, 0.0]      # initial parameter values
  outputs: [8]
  shifts: [0.003, 0.005, 0.007, 0.009]  # EXP perturbations (strings in production)
parameter_settings:
  z:
    bins: 50000
    range: [-25.0, 25.0]
  tprime:
    bins: 1000
    range: [0.0, 1.0]
convergence:
  msd_tol: 1.0e-3
  std_tol: 5.0e-4
```

### QSHE-Specific Config Keys

```yaml
rg_settings:
  metric: "all"   # "mean" | "median" | "std" | "all"
  fixed: 0
data_settings:
  vars: ["r", "t", "tau", "f", "g", "surv", "z", "mix", "p"]
parameter_settings:
  q:
    min: 0.0
    max: 1.0
    num: 50
  p:
    min: 0.0
    max: 1.0
    num: 50
```

---

## Data Formats

### Histogram files (`.npz`)
```
histval     : float64[n_bins]    — histogram counts
binedges    : float64[n_bins+1]  — bin edges
bincenters  : float64[n_bins]    — bin centres
```
Load with: `np.load(file, allow_pickle=False)`

### Sample files (`.npy`)
- 1-D float64 array of amplitude `t` or phase values
- Named: `t_data_RG{step}_{size}_samples.npy`

### Output Directory Layout

**FP (Fixed-Point) run:**
```
{version}/FP/
├── hist/t/t_hist_RG{i}.npz
├── hist/z/z_hist_RG{i}.npz
├── stats/{statistic}_moments.json
└── config/updated_config.yaml
```

**EXP (Shifted) run:**
```
{version}/EXP/shift_{shift}/
├── hist/t/t_hist_RG{i}.npz
└── hist/z/z_hist_unsym_RG{i}.npz
```

---

## Key Variables & Parameters

| Parameter | Typical Values | Description |
|-----------|---------------|-------------|
| `N` | 32M / 320M | Total MC samples |
| `steps` | 7–9 (IQHE), 15 (QSHE) | RG iterations |
| `method` | `"analytic"` / `"numerical"` | RG transformation method |
| `expr` | `"shaw"` | Mathematical expression for RG map |
| `shifts` | `[0.003, 0.005, 0.007, 0.009]` | EXP perturbation values |
| `symmetrise` | 0 / 1 | Whether to symmetrise z-distribution |
| `z_bins` | 50000 | z-histogram resolution |
| `t_bins` | 1000 | t-histogram resolution |
| `seed` | 1234 / 12345 | RNG seed (NumPy PCG64) |
| `matrix_batch_size` | 100000 | Batch size for matrix operations |

### Key Physics Variables

- **`t`** — amplitude (∈ [0, 1]); the fundamental RG variable
- **`z`** — log-ratio form of `t`; symmetric around 0 at critical point
- **`ν`** (nu) — critical exponent extracted from EXP runs
- **`shifts`** — small perturbations applied to the FP distribution to measure ν

---

## IQHE Workflow

### Local Testing (IQHE)

```bash
python -m Local.run_local_iqhe \
  --config Local/configs/local_iqhe \
  --set "rg_settings.steps=3" "rg_settings.samples=10000000" \
  --type FP
# Output: Local data/{version}/FP/
```

### HPC Production Run

```bash
# 1. Push code + scripts + config to cluster
python file_management.py --action push --push code scripts config \
  --version fp_iqhe_numerical_shaw --sys linux

# 2. On the cluster — submit FP run
bash Taskfarm/scripts/run_rg.sh \
  --config Taskfarm/configs/iqhe.yaml \
  --set "engine.method=numerical" \
  --out /tmp/configs

# 3. Submit EXP (shifted) run using FP output
bash Taskfarm/scripts/run_shifts.sh \
  --config Taskfarm/configs/iqhe.yaml \
  --index 0 --out /tmp/configs
```

### Pull Results & Analyse

```bash
# Pull histograms from cluster
python file_management.py --action pull --pull hist \
  --version fp_iqhe_numerical_shaw --type FP --sys linux

# Extract critical exponent
python -m analysis.critical_exponent \
  --version fp_iqhe_numerical_shaw --mode EXP --steps 9
```

### Direct Module CLIs

```bash
# Generate one batch of samples (RG step i)
python -m source.data_generation ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED [EXISTING_T_FILE]

# Build/append histogram
python -m source.histogram_manager PROCESS VAR INPUT_FILE [EXISTING] OUTPUT_FILE RG_STEP [SHIFT]
# PROCESS 0 = initialise, 1 = append

# Launder/symmetrise histograms
python -m source.helpers PROCESS ARRAY_SIZE INPUT OUTPUT SEED
# PROCESS 0 = launder z→t, 1 = symmetrise z, 2 = launder t, 3 = t array→z array

# Validate/update config
python -m source.parse_config --config path.yaml --set "key=val"
```

## QSHE Workflow

The QSHE pipeline is split across two distinct stages with different tooling:

**Data generation (HPC):**
- `source/qshe_data_gen.py` — generates q-p trial data on the Taskfarm
- `source/qshe_data_agg.py` — aggregates trial outputs
- Slurm scripts in `Taskfarm/scripts/` handle job submission
- Outputs land in `Data from taskfarm/` (excluded from context)

**Analysis (local, Jupyter):**
- All post-processing, visualisation, and ν extraction is done in a
  single notebook (not yet refactored into modules)
- The notebook lives at `test_qshe.ipynb`
- Do not expect a CLI equivalent for QSHE analysis — there isn't one
# TODO: refactor into module

When working on QSHE analysis tasks, reference the notebook directly
rather than looking for a `analysis/` equivalent of the IQHE pipeline.

---

## Environment Setup

Required `.env` variables:

```
DATA_DIR    = "...\\Data from taskfarm"    # HPC output destination
LOCAL_DIR   = "...\\Local data"            # Local run output
ROOT_DIR    = "...\\Project Code"          # Repo root
TASKFARM_DIR= "...\\Taskfarm"
QSHE_DIR    = "...\\QSHE"
CONFIG_FILE = "...\\Taskfarm\\configs\\iqhe.yaml"
HOST        = "vulcan2"                    # HPC hostname
USERNAME    = "phuhjf"
REMOTE_DIR  = "/storage/physics/phuhjf/fyp"
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Documentation

Full docs live in [docs/](docs/):

- [docs/Pipeline.md](docs/Pipeline.md) — end-to-end workflow
- [docs/Config.md](docs/Config.md) — all config keys documented
- [docs/Artifacts.md](docs/Artifacts.md) — output file formats
- [docs/Runbook.md](docs/Runbook.md) — operational procedures
- [docs/Troubleshooting.md](docs/Troubleshooting.md) — common issues
