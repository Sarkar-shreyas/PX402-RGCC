# Configuration Reference

The pipeline uses a three-level configuration hierarchy:

1. **YAML config file** — `Taskfarm/configs/iqhe.yaml` (HPC) or
   `Local/configs/local_iqhe.yaml` (local), and `Taskfarm/configs/qshe.yaml` (QSHE HPC).
2. **CLI overrides** — `--set "key.nested.path=value"` flags parsed by
   `source/parse_config.py`.
3. **Validated dataclass** — `IQHEConfig` or `QSHEConfig` (constructed by
   `source/config.py::build_config`).  All downstream modules consume the dataclass,
   not the raw dictionary.

All YAML keys must be lowercase; `_check_lowercase_keys` enforces this on every load
and raises `KeyError` if violated.

---

## IQHE config keys

Reference file: [`Taskfarm/configs/iqhe.yaml`](../Taskfarm/configs/iqhe.yaml)

### `main`

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `version` | string | `"fp_iqhe_numerical_shaw"` | Run identifier; used as the root output directory name |
| `id` | string | `"hpc"` / `"local"` | Environment tag for logs |
| `type` | string | `"fp"` / `"exp"` | Pipeline branch: `fp` = fixed-point run; `exp` = shifted run for ν extraction |
| `output_folder` | string | `""` | Output root path; empty string uses the default from `.env` |

### `engine`

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `model` | string | `"iqhe"` | Physical model: `"iqhe"` or `"qshe"` |
| `method` | string | `"analytic"` / `"numerical"` | RG map implementation: `"analytic"` = closed-form 4-phase map; `"numerical"` = 8-phase matrix (more accurate, ~2× slower) |
| `resample` | string | `"i"` | Resampling strategy: `"i"` = inverse CDF (only supported option) |
| `expr` | string | `"shaw"` | Parametrisation of the RG map; `"shaw"` is the only implemented expression |
| `symmetrise` | int | `1` | Enforce Z₂ symmetry of z-distribution: `0` = off, `1` = on.  Set to `1` at the critical point to suppress finite-sample asymmetry bias |

### `rg_settings`

| Key | Type | Local | HPC | Description |
|-----|------|-------|-----|-------------|
| `seed` | int | `1234` | `12345` | RNG seed (NumPy PCG64); determines the full MC sequence |
| `steps` | int | `7` | `9` | Number of RG iterations |
| `samples` | int | `32000000` | `320000000` | Total MC samples per run |
| `matrix_batch_size` | int | `100000` | `100000` | Samples per matrix-multiply batch; controls peak RAM usage |

### `data_settings`

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `inputs` | list | `[1.0, 0.0]` | Initial distribution parameters `[t₀, phase₀]` |
| `outputs` | list | `[8]` | RG step indices whose histograms are saved to disk |
| `shifts` | list | `[0.003, 0.005, 0.007, 0.009]` | Perturbation magnitudes δ for EXP runs.  **Note:** `constants.SHIFTS` uses string representations `["0.0", "0.003", …]`; YAML values are numeric floats.  Match types carefully when referencing from code |

### `parameter_settings`

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `z.bins` | int | `50000` | Histogram resolution for z = ln(t / (1−t)) |
| `z.range` | list | `[-25.0, 25.0]` | z-axis extent |
| `tprime.bins` | int | `1000` | Histogram resolution for t' ∈ [0, 1] |
| `tprime.range` | list | `[0.0, 1.0]` | Full physical range of t' |

### `convergence`

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `msd_tol` | float | `1.0e-3` | Mean-squared displacement of z-histogram between consecutive RG steps; convergence declared when Δ < threshold |
| `std_tol` | float | `5.0e-4` | Change in std(z) between steps; secondary convergence criterion |

---

## QSHE-specific config keys

Reference file: [`Taskfarm/configs/qshe.yaml`](../Taskfarm/configs/qshe.yaml)

The `main` and `engine` sections are identical in structure to IQHE with
`engine.model = "qshe"`.  Additional and differing keys:

### `rg_settings` (QSHE additions)

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `steps` | int | `15` | Number of RG iterations per (q, p) trial |
| `samples` | int | `100000` | MC samples per (q, p) grid cell |
| `metric` | string | `"all"` | Observable aggregation metric: `"mean"` \| `"median"` \| `"std"` \| `"all"`.  `"all"` stores (median, mean, std) giving `met_dim = 3`; all others give `met_dim = 1`.  **Note:** `"std"` is excluded from the `QSHEConfig` valid values |
| `fixed` | int | `0` | Whether to fix the q parameter during RG iteration: `0` = free, `1` = fixed |

### `data_settings` (QSHE)

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `inputs` | list | `[1.0, 0.0, 0.0, 0.0]` | Initial conditions for the 4-component QSHE state vector |
| `outputs` | list | `["t", "f"]` | Observable names to compute at each RG step |

### `parameter_settings` (QSHE)

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `vars` | list | `["r", "t", "tau", "f", "g", "surv", "z", "mix", "p"]` | Variable names to aggregate in `qshe_data_agg.py` (for local runs) |
| `q.min` | float | `0.0` | Minimum q value across the full grid |
| `q.max` | float | `0.5` | Maximum q value |
| `q.num` | int | `51` | Number of q grid points (current production: 51, spanning [0, 0.5]) |
| `p.min` | float | `0.001` | Minimum p value |
| `p.max` | float | `0.999` | Maximum p value |
| `p.num` | int | `500` | Number of p grid points |

---

## Config invariants

- `rg_settings.samples` must be ≥ `rg_settings.matrix_batch_size`.
- `parameter_settings.z.bins` must be even (enforced by symmetrisation logic).
- `engine.expr` must be `"shaw"` for IQHE (the only implemented expression).
- `engine.model` must be `"iqhe"` or `"qshe"`; `build_config` dispatches on this key.
- `rg_settings.metric` must not be `"std"` for QSHE (`QSHEConfig` validation).
- All keys must be lowercase; `_check_lowercase_keys` is called on every config load.

---

## CLI overrides

Override any nested key at invocation time with `--set`:

```bash
python -m Local.run_local_iqhe \
    --config Local/configs/local_iqhe \
    --set "rg_settings.steps=3" "rg_settings.samples=10000000" \
    --type FP
```

Multiple `--set` arguments are accumulated; quoted `key=value` pairs avoid shell
splitting.  The override is applied after loading the YAML and before validation.

---

## Environment variables (`.env`)

The following variables must be present in the `.env` file at the repository root:

```
DATA_DIR      path to HPC output destination
LOCAL_DIR     path to local run output
ROOT_DIR      path to the repository root
TASKFARM_DIR  path to the Taskfarm directory
HOST          HPC hostname
REMOTE_DIR    remote working directory on the cluster
CONFIG_FILE   path to the default YAML config
```

`constants.py` reads these via `python-dotenv` and raises `RuntimeError` at import
time if any required variable is absent.  `QSHE_DIR` and `QSHE_TEST_DIR` are optional.
