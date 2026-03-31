# CLAUDE.md — RG Monte Carlo Pipeline

## Project Overview

This is a **Renormalization Group (RG) Monte Carlo pipeline** for studying phase transitions in quantum condensed matter systems, specifically:

- **IQHE** (Integer Quantum Hall Effect) — 2D electron systems under strong magnetic fields
- **QSHE** (Quantum Spin Hall Effect) — topological edge states with spin-orbit coupling

The pipeline runs locally for testing (32M samples) and on a Slurm HPC cluster for production (320M samples). It extracts the **critical exponent ν** characterising the divergence of the correlation length at the quantum phase transition.

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
HOST        = # HPC hostname
USERNAME    = # HPC username
REMOTE_DIR  = # HPC remote directory
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

---

## Refactoring Invariants

Everything listed here **must not change** during any cleanup phase.

### Physics Logic

- All RG transformation formulas in `source/utilities.py`: `generate_t_prime` (jack / cain / shaw / t variants), `solve_matrix_eq` (IQHE 10×10 system), `solve_qshe_matrix` (QSHE 20×20 system), and the matrix element assignments in both.
- Variable conversion functions: `convert_t_to_z`, `convert_z_to_t`, `convert_g_to_z`, `convert_z_to_g`, `convert_t_to_g`, `convert_t_to_geff`, `convert_geff_to_t`, `convert_zeff_to_t`, `convert_z_to_x`, `convert_g_to_x`, `convert_x_to_z`, `convert_x_to_g`.
- Symmetrisation: `center_z_distribution` fold-about-zero algorithm.
- Resampling: `inverse_cdf_sampler`, `rejection_sampler`, `launder`, `inverse_cdf_2d`, `rejection_sampler_2d`, `conditional_2d_resampler` — all acceptance/rejection criteria and jitter-within-bin steps.
- ν extraction: `calculate_nu` formula, `estimate_z_peak` bootstrap loop and top-5% bin selection, `fit_z_peaks` R² calculation and slope extraction.
- Convergence metrics: `l2_distance`, `mean_squared_distance`, `hist_moments`.
- QSHE trial logic: `qp_trials`, `run_qp_trials`, `qshe_numerical_solver` batch accumulation, `generate_initial_qshe_data`.
- `generate_initial_t_distribution` (flat g→t draw), `extract_t_samples` (5-column resampling), `generate_random_phases`.

### RNG Seeding

- `build_rng(seed)` must remain `np.random.default_rng(seed=seed)` backed by PCG64.
- Seed values in config (`seed: 1234` local, `seed: 12345` HPC) must not be altered.
- The position of every `build_rng` call relative to other RNG consumers within a function must not change — reordering draws changes the MC sequence.

### Public Interfaces

- CLI positional argument order for all `if __name__ == "__main__"` modules: `data_generation.py`, `histogram_manager.py`, `helpers.py`, `shift_z.py`, `qshe_data_gen.py`, `qshe_data_agg.py` — any change breaks Slurm shell scripts.
- Named CLI flags: `--config`, `--set`, `--type`, `--out`, `--version`, `--steps`, `--loc`, `--mode` across all analysis and driver scripts.
- `build_rng`, `get_rg_config`, `build_config`, `handle_config`, `load_yaml`, `dump_yaml`, `save_updated_config`, `get_nested_data`, `check_required_info` — imported by multiple modules.
- `save_data` NPZ key names (`histval`, `binedges`, `bincenters`) — used by every downstream load and the full Slurm pipeline.
- `rg_data_workflow` dispatch interface; `numerical_t_prime` and `generate_t_prime` signatures.

### Config System

- All YAML key names and their nesting hierarchy (e.g. `engine.method`, `rg_settings.steps`, `parameter_settings.z.bins`) — shell scripts parse these via `get_yaml` helpers and Python modules access them via `get_nested_data`.
- `IQHEConfig` and `QSHEConfig` field names — accessed by attribute throughout the pipeline.
- `build_config` model-dispatch logic (`"iqhe"` / `"qshe"`) and the `"std"` metric exclusion note in `QSHEConfig`.
- `_check_lowercase_keys` validation — silently enforced on every config load.

### Data Schemas

- NPZ histogram schema: three arrays named exactly `histval`, `binedges`, `bincenters` with shapes `(n_bins,)`, `(n_bins+1,)`, `(n_bins,)`. Loaded with `allow_pickle=False` throughout.
- NPY sample array: 1-D float64 array of amplitude values; file-naming convention `t_data_RG{step}_{size}_samples.npy`.
- QSHE NPY output shape: `(len(qs), p_num, num_steps, met_dim)` — relied upon by `qshe_data_agg.py`.
- FP output directory layout: `{version}/FP/hist/t/`, `{version}/FP/hist/z/`, `{version}/FP/hist/sym_z/`.
- EXP output directory layout: `{version}/EXP/shift{s}/hist/`.
- `trial_state.json` schema written by `store_state` — consumed by `test_qshe.ipynb`.

---

## Cleanup Targets

Entries are grouped by file. Each entry lists the function or line range, a one-line description, and a confidence level (high / medium / low) that the change is safe to make without affecting physics outputs or pipeline behaviour.

### `source/utilities.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 34 | `# from collections import defaultdict` — commented-out import at module top | high |
| Line 41 | `# from time import time` — commented-out import at module top | high |
| `solve_qshe_matrix`, line ~782 | `# return ts` — unreachable dead statement after the real `return sol` | high |
| `solve_qshe_matrix`, lines ~745–758 | Large commented-out alternative `b matrix for M` block (superseded by `b matrix for M2`) | high |
| `generate_initial_qshe_data`, lines ~410–418 | Commented-out alternative t-array generation block (8 lines) | high |
| `conditional_2d_resampler`, lines ~1940–1954 | Commented-out alternative rejection-sampler fallback block (13 lines) | high |
| `center_z_distribution` | `z_bins` parameter is accepted but never used; dead parameter | high |
| `qshe_numerical_solver` | Outer loop over `output_indexes` calls `solve_qshe_matrix` (which already processes all indexes) once per index per batch — the 20×20 linear solve runs `len(output_indexes) × num_batches` times instead of `num_batches` times; all but one result is discarded per outer iteration | high |
| `convert_t_to_g` | `np.abs(t) * np.abs(t)` — double call to `np.abs` on a real array; equivalent to `t * t` or `t**2` | medium |
| `convert_t_to_geff` | `np.abs(t) ** 2` and `np.abs(f) ** 2` — redundant `np.abs` on real arrays | medium |
| `convert_geff_to_t` | `np.abs(f) ** 2` — redundant `np.abs` on a real array | medium |
| `build_state_dict` | Function defined in utilities.py but not imported by any active module (superseded by `store_state` in `qshe_data_agg.py`) | medium |
| `get_meds` | Function defined in utilities.py but not imported by any active module visible in the codebase | medium |

### `source/config.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 453 | `# print(f"Key {key} found.")` and `# print(f"Data is currently: {data}")` — commented-out debug prints inside `get_nested_data` | high |
| Line 523 | `# print(overrides)` — commented-out debug print inside `parse_overrides` | high |

### `source/data_generation.py`

| Location | Issue | Confidence |
|---|---|---|
| Lines 137–139 | Commented-out block for deleting the prior-step `existing_t_file` after use | high |

### `source/histogram_manager.py`

No cleanup targets identified beyond what is inherited from imported utilities.

### `source/helpers.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 146 | `# os.remove(input_file)` — commented-out file-cleanup call | high |

### `source/fitters.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 29 | `# from scipy.optimize import curve_fit` — commented-out import | high |
| `_gauss` function (lines 59–61) | Defined but never called anywhere; `curve_fit` usage is also commented out | high |
| `estimate_z_peak`, lines 104–107 | Commented-out mask block (`# Restrict calculations within [-25,25]`) | high |
| `estimate_z_peak`, lines 117, 125, 133–141 | Multiple commented-out `print` and alternative computation lines | high |
| `estimate_z_peak`, lines 109–112 | `np.argsort(z_hist)` called twice on the same array to get `top_indices` and `top_bin_indices`; a single sort with two slice offsets would suffice | medium |
| `fit_z_peaks` | `polynomial.Polynomial.fit` is called to get residuals/R², then `np.polyfit` is called again on the same data to get the slope — two fit calls when one would provide both | medium |

### `source/shift_z.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 77 | `perturbation = shift` — immediate alias with no transformation; `perturbation` and `shift` are used interchangeably, one can be removed | high |

### `analysis/critical_exponent.py`

| Location | Issue | Confidence |
|---|---|---|
| Lines 267–268 | Commented-out `peaks[0, j] = estimate_z_peak(...)` and `means[0, j], stds[0, j] = hist_moments(...)` calls | high |
| Lines 281–283 | Commented-out `test = launder(...)` debug block | high |
| Lines 291–295 | Commented-out alternative y/m computation block ("Without subtracting anything" and "Subtracting the peaks for the Fixed point distribution") | high |
| Lines 315–334 | Second set of commented-out y/m alternatives and duplicate subtraction statements | high |
| Line 433 | `# calculate_average_nu(overall_stats, 7, rg)` — commented-out function call | high |

### `analysis/data_plotting.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 25 | `from mpl_toolkits.mplot3d import Axes3D` — imported but never referenced by name (modern matplotlib registers the 3D projection as a side effect of the import, but the name `Axes3D` itself is unused) | medium |

### `analysis/report_plots.py`

No high-confidence cleanup targets. The `plot_z_fp` index-selection logic (~lines 41–58) is verbose but produces correct report-quality output; low-confidence target only.

### `constants.py`

| Location | Issue | Confidence |
|---|---|---|
| Lines 67–71 | Four commented-out alternative `SHIFTS` definitions accumulated over development | high |
| Lines 85–86 | Two commented-out alternative `CURRENT_VERSION` values (`"1.84J"`, `"1.90S"`) | high |
| Lines 94–95 | Two commented-out alternative `NUM_RG` values (10, 12) | high |

### `Local/run_local_iqhe.py`

| Location | Issue | Confidence |
|---|---|---|
| `rg_fp`, lines ~236–244 | References `args.t` and `args.phi` directly from the enclosing `__main__` scope when `starting_t != 0` / `starting_phi != 0` — creates an implicit dependency on the global `args` object that is undocumented except in the docstring | medium |
| `rg_fp` and `rg_exp` symmetrisation branches | Identical symmetrise / no-symmetrise if/elif/else block duplicated verbatim in both functions; the only difference is which folder variable is used | medium |

### `Local/run_local_qshe.py`

| Location | Issue | Confidence |
|---|---|---|
| `qshe_rg_workflow` FP branch, lines ~322–344 | Large commented-out block of alternative variable assignments (`z_sample`, `g_sample`, `t_sample`, `p_sample`, `f_sample_conv`) and an `assert` block | high |
| `save_hist`, line 675 | `np.savez_compressed(filename, **data, allow_pickle=True)` — inconsistent with the rest of the pipeline which loads NPZ files with `allow_pickle=False`; the `allow_pickle=True` argument to `savez_compressed` is a no-op (savez never pickles), but the inconsistency is misleading | medium |

### `file_management.py`

| Location | Issue | Confidence |
|---|---|---|
| Line 285 | `elif str(args.step).isdigit:` — missing parentheses; `str.isdigit` is a method object (always truthy), so the `else: raise ValueError` branch on line 288 is unreachable dead code | high |
| Lines 340–342 | `commands = ["scp", "-r"]` reset at the end of the push/pull loop body — resets to scp even when `args.sys` is `"linux"`/`"mac"` (which originally set rsync); second and subsequent targets always use scp regardless of OS | medium |
