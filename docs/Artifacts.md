# Artifacts

This document describes every file format and directory layout produced by the pipeline.
For a description of how each artifact is generated, see [Pipeline.md](Pipeline.md).

---

## Shared file formats

### NPZ histogram (`.npz`)

All histogram files use this schema:

| Key | dtype | Shape | Description |
|-----|-------|-------|-------------|
| `histval` | float64 | `(n_bins,)` | Raw histogram counts |
| `binedges` | float64 | `(n_bins + 1,)` | Bin edge positions |
| `bincenters` | float64 | `(n_bins,)` | Bin centre positions |

Load with:

```python
data = np.load(filepath, allow_pickle=False)
counts   = data["histval"]
edges    = data["binedges"]
centers  = data["bincenters"]
```

**`allow_pickle=False` is mandatory** throughout the pipeline.

### NPY sample array (`.npy`)

1-D float64 array of amplitude values.  Naming convention:

```
t_data_RG{step}_{size}_samples.npy
```

---

## IQHE output directory layout

### FP run

```
{version}/FP/
├── hist/
│   ├── t/
│   │   └── t_hist_RG{i}.npz            t-histogram at step i
│   ├── z/
│   │   └── z_hist_RG{i}.npz            z-histogram at step i (unsymmetrised)
│   └── sym_z/
│       └── sym_z_hist_RG{i}.npz        symmetrised z-histogram (when engine.symmetrise=1)
├── stats/
│   └── {statistic}_moments.json        convergence statistics per RG step
└── config/
    └── updated_config.yaml             resolved config snapshot for this run
```

### EXP run

```
{version}/EXP/shift{s}/hist/
├── t/
│   └── t_hist_RG{i}.npz
└── z/
    └── z_hist_unsym_RG{i}.npz
```

`{s}` is the shift value (e.g. `0.003`).

### Analysis outputs (`analysis/critical_exponent.py`)

```
{data_folder}/{version}/
├── peaks.json               per-(RG step, shift) Gaussian peak estimates
├── overall_stats.json       per-RG-step ν, slope, R² from peak and mean fits
├── z_peaks.png              scatter/line plot of peak displacement vs shift
├── Nu_{N}_shifts.png        ν vs system size (2^n) with error bars
└── EXP/
    ├── stats/{shift}/       per-shift moment statistics
    └── plots/{shift}/       per-shift histogram PNG files
```

---

## QSHE output directory layout

### Per-block generation (`source/qshe_data_gen.py`)

Written to a local scratch directory during the job, then moved atomically to the
shared filesystem:

```
{OUTPUT_DIR}/q{b}/
├── p_data_q{b}_{N}_samples.npy   shape (q_block_size, p_num, num_steps, met_dim)
├── q_data_q{b}_{N}_samples.npy   shape (q_block_size, p_num, num_steps, met_dim)
└── DONE                           empty sentinel; must exist before aggregation
```

### Aggregated data (`source/qshe_data_agg.py`)

```
{OUTPUT_DIR}/
├── p_data_agg.npy                 shape (q_num, p_num, num_steps, met_dim)
├── q_data_agg.npy                 shape (q_num, p_num, num_steps, met_dim)
└── trial_state.json               grid metadata and run parameters
```

**Array shape** for the current production config (`Taskfarm/configs/qshe.yaml`):
`(51, 500, 15, 3)` — (q_num=51, p_num=500, steps=15, met_dim=3 for `metric="all"`).

`met_dim = 3` stores (median, mean, std) per observable per step.
`met_dim = 1` for any other metric value.

### `trial_state.json` schema

Written by `source/qshe_data_agg.py::store_state`:

```json
{
  "q":      {"Min": float, "Max": float, "Num": int, "Step": float},
  "p":      {"Min": float, "Max": float, "Num": int, "Step": float},
  "vars":   ["p", "q"],
  "config": {
    "Samples":    int,
    "Steps":      int,
    "Fixed":      bool,
    "Metric":     "all",
    "Block size": int
  }
}
```

The notebook (`test_qshe.ipynb`) does **not** read `trial_state.json` directly; it
reconstructs the grid from `updated_config.yaml` via `build_config`.  The JSON is
provided as a human-readable record.

### Updated config snapshot

```
{DATA_DIR}/{dataversion}/QP/config/updated_config.yaml
```

Written by `source/config.py::save_updated_config`.  This is the resolved config
(YAML + CLI overrides applied) and is the file that the notebook loads.

### Notebook plot outputs

All plots are saved under `{DATA_DIR}/{dataversion}/QP/plots/`:

```
plots/
├── vfield/p/          streamplots and velocity heatmaps for p
├── vfield/q/          streamplots and velocity heatmaps for q
├── vfield/pq/         combined speed heatmaps
├── grid/              gridded RG flow trajectory plots
├── boundary/          phase boundary plots
├── density/           parameter density histograms
├── Gamma/             Γ(p) crossing and p_c(q) plots
├── Nu/                ν vs q and ν vs system size plots
├── stds/              per-(q,p) standard deviation plots
└── variance/          per-(q,p) variance plots
```

Report-quality figures are saved directly to `./report/`:

```
report/
├── nu_qshe.pdf          ν vs q_init for selected fitting windows
├── qshe_nu_q0.pdf       ν vs RG step for q = 0 with literature comparison
├── fitted_pc.pdf        fitted p_c vs q_init
├── logfit.pdf           ln|T_k| vs ln(2^k) scaling plot
└── z_pc.pdf             z(p) curves at p_c for multiple RG steps
```

---

## Cluster artefact locations

Remote artefacts live under `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/`.  Pull them
with `file_management.py`:

```bash
python file_management.py --action pull --pull hist \
    --version VERSION --type FP --sys linux
```

Slurm job logs are written to `<REMOTE_ROOT>/job_logs/{version}/{FP|EXP}/`.
