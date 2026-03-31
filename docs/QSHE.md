# QSHE Analysis

This document covers the Quantum Spin Hall Effect (QSHE) analysis pipeline
end-to-end.  It is written to be self-contained: a reader should be able to understand
the QSHE analysis without consulting any other document in this folder.

---

## Table of contents

1. [Physical context](#1-physical-context)
2. [Data inputs](#2-data-inputs)
3. [Notebook structure](#3-notebook-structure)
4. [Key outputs](#4-key-outputs)
5. [How to run](#5-how-to-run)
6. [Relationship to the IQHE pipeline](#6-relationship-to-the-iqhe-pipeline)

---

## 1. Physical context

### The Quantum Spin Hall Effect

The Quantum Spin Hall Effect (QSHE) is a time-reversal-invariant topological phase of
matter driven by spin-orbit coupling.  In contrast to the Integer Quantum Hall Effect
(IQHE), the QSHE does not require a magnetic field; instead, spin-orbit coupling acts
as an effective field of opposite sign for opposite spins, producing counter-propagating
spin-polarised edge states protected by time-reversal symmetry.  The topological
invariant classifying QSHE phases is ℤ₂: the system is either in a topological
insulator phase (with protected helical edge states) or a trivial insulator.

At the quantum phase transition separating these phases, the correlation length ξ
diverges as:

    ξ ~ |δ|^{−ν}

where δ is the detuning from the critical point and ν is the critical exponent
characterising the universality class.  The universality class of the QSHE transition
is expected to differ from the IQHE (Chalker–Coddington) universality class due to the
additional time-reversal symmetry constraint.  The literature value for the QSHE
critical exponent from Kobayashi et al. (2010) is ν ≈ 2.73.

### The ℤ₂ network model

The QSHE network model used here is a ℤ₂ analogue of the Chalker–Coddington (CC)
network model.  In the CC model, electrons drift along iso-potential contours of a
smooth disordered potential landscape and tunnel quantum-mechanically at saddle-point
nodes.  The scattering at each node is described by a unitary S-matrix parameterised
by a transmission amplitude t.

The QSHE generalisation introduces a second amplitude f that couples the two spin
channels (the "spin-flip" amplitude), alongside the spin-conserving amplitude t.  A
node is therefore characterised by the pair (t, f), or equivalently by the
dimensionless parameters:

    p = t²          (spin-conserving transmission probability)
    q = f² / (1−p)  (conditional spin-flip probability)

Both p and q lie in [0, 1].  The special case q = 0 reduces to the IQHE; q > 0
introduces spin-mixing and drives the system towards the QSHE universality class.

This parametrisation follows Kobayashi, Ohtsuki, Obuse, and Slevin,
*Phys. Rev. B* **82**, 165301 (2010).

### The RG cell and RG flow

An RG cell combines five scattering nodes into a single renormalised node.  The RG
transformation maps (t, f) → (t', f') by sampling five input amplitudes from a
distribution and computing the output amplitude via the QSHE matrix equation
(`source/utilities.py::solve_qshe_matrix`, a 20×20 linear system).

Iterating the RG transformation across a (p, q) grid traces out the RG flow.  The
flow is computed independently for each (q, p) initial condition and recorded at every
RG step.  The result is a trajectory in (p, q) space that converges towards a fixed
point, a phase boundary, or a phase sink depending on the initial condition.

The quantity recorded at each step is configurable via `rg_settings.metric`; with
`metric = "all"` (the production setting) all three of (median, mean, std) are stored
for both p and q.

### What the notebook extracts

The notebook extracts the critical exponent ν using a finite-size scaling (FSS)
analysis of the RG flow.  The approach mirrors the method of Kobayashi et al. (2010):

1. Define Γ_n(p) = z_n(p) / 2^(n+1) where z = ln((1−g)/g), g = p + (1−p)q, and n is
   the RG step index (system size L_n = 2^(n+1)).
2. Identify the critical point p_c(q) as the crossing of consecutive Γ_n(p) curves.
3. Compute the slope T_k(q) = dΓ/dp |_{p = p_c(q)} at RG step k.
4. The FSS hypothesis predicts T_k ~ L_k^{1/ν}, so a log-log fit of |T_k| vs 2^k
   gives slope m and hence ν = 1/(m + 1).

An effective ν is also computed step-by-step as:

    ν_eff = 1 / (Δ_slope / ln 2 + 1)

where Δ_slope is the difference in log-slope between consecutive steps.

---

## 2. Data inputs

### Source of the data

Data is produced by the two-stage HPC pipeline:

1. **`source/qshe_data_gen.py`** — runs on the Slurm cluster as a job array (one
   element per q-block).  For each (q, p) grid cell it calls
   `source/utilities.py::qp_trials`, which runs `num_samples` Monte Carlo trials
   through `num_steps` RG iterations.  Outputs are written to per-block `.npy` files
   with a `DONE` sentinel.

2. **`source/qshe_data_agg.py`** — checks all `DONE` sentinels and concatenates the
   per-block files into two aggregated arrays, then writes `trial_state.json`.

3. The aggregated files land in `{DATA_DIR}/{dataversion}/QP/data/` and are pulled to
   the local machine with `file_management.py`.

See [Pipeline.md](Pipeline.md) and [Runbook.md](Runbook.md) for the generation and
aggregation commands.

### File locations

The notebook reads data from three paths derived from the `dataversion` variable and
the `DATA_DIR` environment variable (loaded from `.env` via `constants.py`):

```
{DATA_DIR}/{dataversion}/QP/data/p_data_agg.npy
{DATA_DIR}/{dataversion}/QP/data/q_data_agg.npy
{DATA_DIR}/{dataversion}/QP/config/updated_config.yaml
```

The current production `dataversion` is hardcoded in `test_qshe.py` at line 977:

```python
dataversion = "qp_unfixed_numerical_shreyas"
```

To analyse a different dataset, update this string before running the notebook.

### Array shapes and format

Both aggregated files are uncompressed NumPy `.npy` files, dtype `float64`:

| File | Shape | Axes |
|------|-------|------|
| `p_data_agg.npy` | `(q_num, p_num, num_steps, met_dim)` | q-index, p-index, step, metric |
| `q_data_agg.npy` | `(q_num, p_num, num_steps, met_dim)` | q-index, p-index, step, metric |

For the current production config (`Taskfarm/configs/qshe.yaml`) the shape is
`(51, 500, 15, 3)`:

- `q_num = 51`, q ∈ [0.0, 0.5] uniformly spaced
- `p_num = 500`, p ∈ [0.001, 0.999] uniformly spaced
- `num_steps = 15` RG iterations
- `met_dim = 3` because `metric = "all"` stores (median, mean, std) in axis 3

Metric axis layout (index 3): `[0] = median`, `[1] = mean`, `[2] = std`.

### `trial_state.json` schema

Written by `source/qshe_data_agg.py::store_state` alongside the aggregated data:

```json
{
  "q":      {"Min": 0.0, "Max": 0.5, "Num": 51, "Step": 0.01},
  "p":      {"Min": 0.001, "Max": 0.999, "Num": 500, "Step": 0.002},
  "vars":   ["p", "q"],
  "config": {
    "Samples": 100000,
    "Steps": 15,
    "Fixed": false,
    "Metric": "all",
    "Block size": 10
  }
}
```

The notebook reconstructs the q and p grids from `updated_config.yaml` via
`build_config` (not from `trial_state.json` directly); the JSON is a
human-readable record.

---

## 3. Notebook structure

The notebook is `test_qshe.ipynb`; its nbconvert export is `test_qshe.py`, which
is kept in sync and serves as the authoritative readable source.  The description below
uses cell boundary markers and heading comments from that export.

### 3.1 Initialisation (line 43)

**What it does:** imports, RNG seeding, module setup.

Imports `numpy`, `matplotlib`, `scipy`, `json`, `os`, and the following project
modules via wildcard imports:

- `from constants import *` — path variables (`data_dir`, `local_dir`, `root_dir`, …)
  and plot-parameter dicts (`XLIMS`, `YLIMS`, `LEGENDS`)
- `from source.utilities import *` — RG math functions including `build_rng`,
  `solve_qshe_matrix`, `convert_g_to_z`, `generate_random_phases`
- `from source.config import *` — `build_config`, `load_yaml`, `QSHEConfig`
- `from QSHE.testing_qshe import *` — auxiliary QSHE testing utilities
- `from Local.run_local_qshe import *` — local QSHE driver; provides
  `build_default_output_dir` for constructing the default output path

The RNG is seeded once: `rng = build_rng(1234)`.

**Source functions called:** `build_rng` (`source/utilities.py`)

**Intermediate results:** global `rng` object; all constants and utility functions
available in the notebook namespace.

### 3.2 Helper functions (line 71)

**What it does:** defines notebook-local helper functions used throughout later
sections.

Key helpers:

| Function | Purpose |
|----------|---------|
| `find_intersections` | Compute per-x variance of y-values across RG steps; used to locate fixed-point crossing candidates |
| `close_old_plots` | Close all open matplotlib figures |
| `load_var_moments` | Load FP moments from archived local theta-test data (legacy helper) |
| `load_data` | Load aggregated `p_data_agg.npy` / `q_data_agg.npy` from `DATA_DIR`; NaN-fill missing values; compute RG velocity fields (`pvel`, `qvel`) as step-to-step differences; reconstruct p and q grids from config |
| `plot_stream` | Streamplot of (p_vel, q_vel) over the (p, q) grid with optional contours at zero-velocity |
| `speed_heatmap` | 3-panel heatmap of velocity magnitude over consecutive RG steps |
| `find_peaks` | Identify candidate fixed points by thresholding on flow speed |
| `get_jacobian_and_eigenvals` | Compute discrete Jacobian of the RG map at a candidate fixed point using `numpy.gradient`; return eigenvalues |
| `get_boundary` | Label each (p, q) grid point as phase 0 (low), phase 1 (high), or undecided (2) based on the final-step p value; detect phase boundary cells |
| `plot_boundaries` | Plot phase boundary lines across all RG steps, optionally smoothed via B-spline fitting |
| `plot_densities` | Histograms of a chosen variable at each RG step |
| `plot_gammas` | Plot Γ_n(p) curves for a given q index and step range |
| `crossing_p` | Find the first zero-crossing of y₁(p) − y₂(p) by linear interpolation |
| `plot_crossings` | Compute and plot p_c(q) for consecutive pairs of RG steps |
| `gamma_slope` | Fit a degree-2 polynomial to Γ near p_c and return the linear coefficient (slope at p_c) |
| `fit_nu` | Least-squares fit of ln|T_k| vs ln(2^k); returns ν = 1/m |
| `grid_coords` | Scatter-and-quiver plot of RG trajectory between two steps |

**Source functions called:** `convert_g_to_z` (`source/utilities.py`);
`source/fitters.py::fit_z_peaks` (called later in the ν section)

### 3.3 Post-Taskfarm implementation (line 945)

**What it does:** initialises plot-style variables, creates output subdirectories, and
calls `load_data`.

```python
dataversion = "qp_unfixed_numerical_shreyas"
dataconfig  = load_yaml(f"{data_dir}/{dataversion}/QP/config/updated_config.yaml")
plotdir     = f"{data_dir}/{dataversion}/QP/plots"
cfg         = build_config(dataconfig)   # → QSHEConfig
pvel, qvel, gs, qs, numsteps, pdata, qdata = load_data(cfg, dataversion)
```

Derived quantities computed here:

```python
gdata = pdata + (1 - pdata)*qdata          # conductance g = p + (1−p)*q
zdata = convert_g_to_z(gdata)              # z = ln((1−g)/g)
tdata = np.sqrt(pdata)                      # amplitude t
fdata = np.sqrt(qdata*(1 - pdata))         # spin-flip amplitude f
```

The `vardatadict` dictionary maps variable names to their data arrays:
`{"t": tdata, "f": fdata, "p": pdata, "q": qdata, "g": gdata, "z": zdata}`.

**Source functions called:** `load_yaml`, `build_config` (`source/config.py`);
`convert_g_to_z` (`source/utilities.py`)

**Intermediate results:** `pdata`, `qdata` (shape `(51, 500, 15, 3)`), all derived
arrays, `gs` (p-axis, shape `(500,)`), `qs` (q-axis, shape `(51,)`).

### 3.4 Taskfarm data analysis (line 965)

**What it does:** additional setup; computes per-step standard deviations from
`pdata[:, :, :, 2]` (the std metric slice); defines `plot_mets` for std/variance
visualisation.

**Intermediate results:** `pstd`, `qstd` arrays of shape `(51, 500, 15)`.

### 3.5 Field analysis — Streamplots (line 1045)

**What it does:** iterates over all consecutive RG step pairs and saves a velocity
streamplot for each using `plot_stream`.

```python
for i in range(numsteps-1):
    p_vel = pvel[:, :, i, 0]   # median velocity (metric index 0)
    q_vel = qvel[:, :, i, 0]
    plot_stream(gs, qs, i, i+1, p_vel, q_vel, streamplotdir, ...)
```

**Outputs:** PNG files saved to `{plotdir}/vfield/flow_diag_{i}to{i+1}.png`.

### 3.6 Check grid flow (line 1087)

**What it does:** plots scatter-and-quiver diagrams of RG trajectory points for a
subsampled grid, illustrating how initial (p, q) values evolve over a fixed number of
steps.

**Outputs:** PNG files saved to `{plotdir}/grid/`.

### 3.7 Checking phases — Velocity heatmap (line 1099)

**What it does:** classifies grid points by their final-step p value (phase 0 / 1 /
undecided); computes per-step velocity increments `del_p`, `del_q`, `del_speed`; saves
speed heatmaps for all consecutive step triplets.

**Intermediate results:** `del_p`, `del_q`, `del_speed` (shape `(51, 500, 14, 3)`);
`p_ins0`, `p_ins1` masks; `arc_pq` arc-length array.

**Outputs:**
- Phase scatter plot displayed inline.
- Speed heatmap PNGs saved to `{plotdir}/vfield/{vel}/`.

### 3.8 Check candidate FPs and eigenvalues (line 1157)

**What it does:** calls `find_peaks` to identify (p, q) positions with the lowest RG
flow speed (candidate fixed points), then computes the discrete Jacobian and eigenvalues
at each candidate.

**Source functions called:** `find_peaks`, `get_jacobian_and_eigenvals` (local helpers)

**Intermediate results:** `peaks` list of `(p, q, density)` tuples; eigenvalue arrays
printed to stdout.

<!-- TODO: verify against notebook — the stepnum inside find_peaks is hardcoded to 2 regardless of the argument passed -->

### 3.9 Boundary plots (line 1170)

**What it does:** calls `plot_boundaries` to draw phase boundary lines for every other
RG step, optionally smoothed with B-spline fitting.

**Source functions called:** `get_boundary`, `make_smooth_line` (local helpers);
`make_splprep` from `scipy.interpolate`

**Outputs:** PNG saved to `{plotdir}/boundary/boundaries_{lb}to{ub}.png`.

### 3.10 Density plots (line 1185)

**What it does:** iterates over all variables in `vardatadict` and saves a density
histogram for each.

**Outputs:** PNG files saved to `{plotdir}/density/`.

### 3.11 Gamma Analysis (line 1203)

**What it does:** this section implements the core ν extraction machinery.  It has
multiple sub-cells:

**Γ construction:** computes Γ_n(p) = z_n(p) / 2^(n+1) for all (q, p) at each step n.
`zdata[:, :, :, 0]` (median z) is used.

```python
gammas[j, :, k] = zdata[j, :, k, 0] / 2**(k+1)
```

**Critical-point finding:** calls `plot_crossings` to compute p_c(q) for each
consecutive pair of steps.  The result is stored in `crossings` with shape
`(qgamma.size, num_step_pairs)`.  The analysis is restricted to
q ∈ [0, 0.5] and p ∈ (0.3, 0.95) to focus on the physically relevant region.

**Fitting p_c:** for each q, `fits` is computed by linear regression of
p_c^{(k)}(q) against the system size 2^k.  The intercept gives the extrapolated p_c
and is stored in `fits` with shape `(qgamma.size, endcrossing − crossingstep − 2)`.

```python
pc = np.mean(fits[:, :-1], axis=1)   # mean p_c across fitting windows
```

**Slope computation:** `gamma_slope` is called for each (q, k) pair to compute
T_k(q) = dΓ/dp |_{p_c(q)}.  Results are stored in `slopes` with shape
`(qgamma.size, endcrossing − crossingstep − 1)`.

**Intermediate results:** `gammas` (shape `(51, 500, 15)`), `crossings`, `fits`,
`slopes`, `gammar2s`, `gammaerrs`.

### 3.12 Nu / Critical exponent (line — several sub-cells after Gamma Analysis)

**What it does:** computes ν from the slopes using two complementary methods.

**Method A — global fit (ν vs q):**
For each q, fits ln|T_k| vs ln(2^k) over a window of steps using `numpy.polyfit`
(degree 1).  The slope m gives ν = 1/(m + 1).  Results are collected across all
starting-step / ending-step windows and stored in `nutests`, `meannus`, `stdnus`.
The main ν-vs-q plot is saved to `./report/nu_qshe.pdf`.

**Method B — effective ν:**
Computes step-to-step effective ν:

```python
slopediffs = logslope[:, 1:] - logslope[:, :-1]
nueff = 1 / ((slopediffs / np.log(2)) + 1)
```

`nueff` has shape `(qgamma.size, num_steps − 1)`.  Plotted vs RG step k for all q
values; saved to `./report/nu_eff_allq.pdf`.

**Method C — z-peak scaling (used in `get_nus`):**
For a chosen q, selects `choicepsep` p-values near p_c, fits a Gaussian to the z
distribution at each (p, step), and fits the mean z-value vs a z₀ axis using
`source/fitters.py::fit_z_peaks`.  The slope s gives ν = ln(2^(i+1)) / ln(s).
Results are collected in `mets` with shape `(qgamma.size, endstep, 4)` where axis 2
stores (slope, R², ν, slope error).

**Source functions called:** `fit_z_peaks` (`source/fitters.py`);
`scipy.stats.norm.fit`

**Outputs:**
- `{plotdir}/Nu/nu_vs_q.png` — ν vs q_init
- `{plotdir}/Nu/nu_vs_system_size_q{q}.png` — ν vs system size for a chosen q
- `./report/nu_qshe.pdf` — ν vs q_init (report quality)
- `./report/qshe_nu_q0.pdf` — ν vs RG step for q = 0 with literature reference lines
- `./report/fitted_pc.pdf` — fitted p_c vs q_init
- `./report/logfit.pdf` — ln|T_k| vs ln(2^k) scaling plot
- `./report/qcut.pdf` — ν̄ vs q_cut for good fitting windows

### 3.13 Disorder potential (line 2454)

**What it does:** generates a synthetic 2D disordered potential landscape for the
Chalker–Coddington model using Gaussian smoothing of white noise, identifies saddle
points (critical nodes) by Hessian classification, and produces a publication-quality
figure.

This section is self-contained and does not depend on the loaded QSHE data.  It uses
`scipy.ndimage.gaussian_filter` for potential generation and
`scipy.signal.argrelextrema` for critical-point detection.

**Outputs:** figure displayed inline; optionally saved to `./report/chalker_coddington.pdf`
(save line is commented out).

### 3.14 Landau and Conductance

<!-- TODO: verify against notebook — this section appears to be empty or contains only report-specific plotting code -->

### 3.15 Deprecated (line 1861)

**What it does:** archived analysis code, no longer in active use.  Contains
alternative ν extraction approaches using `fit_nu` (a different fitting convention).
These cells should not be run as part of normal analysis.

---

## 4. Key outputs

| Output | Location | Description |
|--------|----------|-------------|
| `flow_diag_{i}to{j}.png` | `{plotdir}/vfield/` | Streamplots of RG velocity field |
| Speed heatmaps | `{plotdir}/vfield/{p,q,pq}/` | Heatmaps of \|dp/dn\|, \|dq/dn\|, combined speed |
| Grid flow plots | `{plotdir}/grid/` | Trajectory scatter-quiver plots |
| Boundary plots | `{plotdir}/boundary/` | Phase boundary lines across RG steps |
| Density histograms | `{plotdir}/density/` | Variable distributions at each step |
| Γ(p) crossing plots | `{plotdir}/Gamma/` | p_c(q) per step pair |
| `nu_vs_q.png` | `{plotdir}/Nu/` | ν vs q_init |
| `nu_vs_system_size_q*.png` | `{plotdir}/Nu/` | ν vs system size for a chosen q |
| `nu_qshe.pdf` | `./report/` | ν vs q_init (report quality) |
| `qshe_nu_q0.pdf` | `./report/` | ν vs RG step for q = 0 with literature references |
| `fitted_pc.pdf` | `./report/` | Fitted p_c vs q_init |
| `logfit.pdf` | `./report/` | ln\|T_k\| vs ln(2^k) |
| `qcut.pdf` | `./report/` | ν̄ vs q_cut for all valid fitting windows |

---

## 5. How to run

### Local data generation

Use `Local/run_local_qshe.py` to run a full (q, p) grid sweep on a single machine
without Slurm.  The output format is identical to the HPC aggregated data and can be
read directly by the notebook.

Run from the repository root:

```bash
python -m Local.run_local_qshe \
    --config Local/configs/local_qshe_qp
```

For a quick smoke-test with a coarse grid and reduced sample count:

```bash
python -m Local.run_local_qshe \
    --config Local/configs/local_qshe_qp \
    --set "rg_settings.samples=10000" "rg_settings.steps=5" \
           "parameter_settings.q.num=10" "parameter_settings.p.num=10"
```

Outputs are written to:

```
Local data/{version}_{method}_{expr}/QP/
    config/updated_config.yaml
    data/p_data_agg.npy          shape (q_num, p_num, steps, met_dim)
    data/q_data_agg.npy          shape (q_num, p_num, steps, met_dim)
    data/trial_state.json
    output.txt
    error.txt
```

To analyse this local run in the notebook, set `DATA_DIR` in `.env` to
`<repo root>/Local data/` and set `dataversion` to `{version}_{method}_{expr}`
(e.g. `"rg_qshe_numerical_shreyas"`).

---

### Environment requirements

The notebook requires the same Python environment as the rest of the pipeline:

```
numpy >= 1.26
scipy >= 1.11
matplotlib >= 3.8
PyYAML >= 6.0
```

Install with:

```bash
pip install -r requirements.txt
```

A valid `.env` file must exist at the repository root with at least:

```
DATA_DIR  = "...\Data from taskfarm"
LOCAL_DIR = "...\Local data"
ROOT_DIR  = "...\Project Code"
```

`constants.py` will raise `RuntimeError` at import time if any of the required
variables (`DATA_DIR`, `LOCAL_DIR`, `ROOT_DIR`, `TASKFARM_DIR`, `HOST`, `REMOTE_DIR`,
`CONFIG_FILE`) is absent.

### Pointing the notebook at a dataset

1. Ensure the aggregated data files exist at
   `{DATA_DIR}/qp_unfixed_numerical_shreyas/QP/data/p_data_agg.npy` and
   `q_data_agg.npy`.
2. If using a different dataset, update the `dataversion` string at line 977 of
   `test_qshe.py` (or the equivalent cell in `test_qshe.ipynb`):
   ```python
   dataversion = "your_dataversion_string"
   ```
3. Ensure `{DATA_DIR}/{dataversion}/QP/config/updated_config.yaml` exists.  This is
   written by `source/config.py::save_updated_config` during the HPC aggregation step.

### Running the notebook

Open `test_qshe.ipynb` in Jupyter and run all cells top-to-bottom.  The notebook is
**not safe to run out of order**: later cells depend on variables (`fits`, `crossings`,
`slopes`, `logsize`, etc.) defined in earlier cells.  Use **Kernel → Restart and Run
All** for a clean execution.

### Known gotchas

- **Hardcoded `dataversion`** (line 977): must be updated manually to point to a
  different dataset.
- **`plotdir` is derived from `dataversion`** (line 979): if the data directory does
  not exist the `os.makedirs` calls will create it, but the NPY loads will still fail.
- **`fits` referenced before definition in some cells** (e.g. line 1237 references
  `fits` which is first defined at line 1371): the notebook must be run from the
  beginning; do not run these cells in isolation.
- **`logsize` referenced at line 1484 before defined at line 1548**: same constraint —
  run all cells in order.
- **`quicker_trials` function uses a module-level `rng`** (lines 215–275): this
  function captures the global `rng` from the Initialisation section.  If the notebook
  is re-run without a full kernel restart, `rng` may be in a different state.
- **Report paths are hardcoded** as `./report/...`: these are relative to the working
  directory at notebook launch, not to `ROOT_DIR`.  Ensure the notebook is launched
  from the repository root.
- **`QSHE_TEST_DIR` is optional**: the `.env` variable `QSHE_TEST_DIR` is not in the
  required list in `constants.py` and will silently be `None` if absent.

---

## 6. Relationship to the IQHE pipeline

### What is shared

| Component | Shared by both pipelines |
|-----------|--------------------------|
| `source/utilities.py` | Core RG mathematics, variable conversions (`convert_g_to_z`, `convert_z_to_t`, …), `build_rng`, `solve_qshe_matrix`, `qp_trials` |
| `source/config.py` | Config loading, YAML I/O, `build_config`, `QSHEConfig` dataclass |
| `constants.py` | Path variables, `XLIMS`, `YLIMS`, `LEGENDS` |
| Config system | Three-level hierarchy (YAML → `--set` overrides → validated dataclass) |
| `.env` / environment setup | Same variables required; same `load_dotenv` mechanism |

### What is structurally different

| Aspect | IQHE | QSHE |
|--------|------|------|
| Analysis driver | `Local/run_local_iqhe.py` (CLI), `analysis/critical_exponent.py` (CLI) | `test_qshe.ipynb` (Jupyter notebook only) |
| Data representation | 1-D histograms of t and z, resampled at each step | 4-D arrays `(q, p, step, metric)` over a parameter grid |
| ν extraction method | RSRG perturbation: FP + EXP runs, log-slope of peak displacement | FSS of RG flow: gamma-crossing analysis, log-slope of Γ-gradient |
| HPC parallelism | One Slurm array per RG step (aggregate into histograms) | One Slurm array per q-block (aggregate into arrays) |
| Config dataclass | `IQHEConfig` | `QSHEConfig` |
| Production steps | 9 RG steps | 15 RG steps |
| Output format | NPZ histograms | NPY float64 arrays |
| Post-processing | Fully CLI-driven (`analysis/critical_exponent.py`) | Notebook only; no CLI equivalent |
