**Deployment / Staging model (summary)**

- Local repo layout: `<LOCAL_REPO_ROOT>` (your git checkout). Key folders: `source/`, `Taskfarm/scripts/`, `Taskfarm/configs/`, `Local/`.
- Remote staged layout: code and runtime assets are uploaded to `<REMOTE_ROOT>` on `<HOST>` by `file_management.py`.

- Mapping (authoritative):
	- Local `source/` → Remote `<REMOTE_ROOT>/code/source/`
	- Local `Taskfarm/scripts/*.sh` → Remote `<REMOTE_ROOT>/scripts/`
	- Local `Taskfarm/configs/*` → Remote `<REMOTE_ROOT>/scripts/`

All remote references below assume the staged layout above. `file_management.py` is the source of truth for pushes/pulls.

---

**Pipeline overview**

This document describes the runtime lifecycle for a single RG step when running on the cluster (staged scripts under `<REMOTE_ROOT>/scripts/`).

1) Master submission

- The master orchestration script on the cluster is `<REMOTE_ROOT>/scripts/rg_fp_master.sh` (FP) or `<REMOTE_ROOT>/scripts/shifted_rg.sh` (EXP). These masters submit worker jobs and use `sbatch --dependency` to enforce ordering.

Example (submit master job from the cluster):

```bash
sbatch --parsable <REMOTE_ROOT>/scripts/rg_fp_master.sh /path/to/updated_config.yaml
```

2) Generation array jobs

- The master submits generation array jobs that run `<REMOTE_ROOT>/scripts/rg_gen_batch.sh`. Each array task executes `python -m source.data_generation` using the staged code under `<REMOTE_ROOT>/code/source/`.
- Each successful batch is rsynced to shared storage and a `READY` marker is written for aggregation.

3) Histogram aggregation

- The histogram manager script `<REMOTE_ROOT>/scripts/rg_hist_manager.sh` waits for `READY` markers, converts t'→z (`source/helpers.py`), and aggregates histograms via `source/histogram_manager.py`.
- NPZ histograms are written to the shared job outputs area. Laundering/symmetrisation is performed using functions in `source/utilities.py`.

4) Iteration and outputs

- Laundered t batches are produced for the next RG step by the histogram manager; the master schedules the next step until the configured number of RG steps completes.

---

Notes & references

- The staged runtime code path on the cluster is `<REMOTE_ROOT>/code/source/` (push implemented in `file_management.py`).
- The cluster script locations are `<REMOTE_ROOT>/scripts/` (push implemented in `file_management.py`).
- For exact transfer behaviour and examples, see `file_management.py` in the repo root.

---
# Expanded details
This pipeline is designed to enable large-scale Monte-Carlo RG steps across Slurm. Data is split into manageable batches and aggregated per step to use histogram-only handoffs to allow samples on the order of 10^8 and greater.

Big-picture diagram:

```text
<REMOTE_ROOT>/scripts
 └─ rg_fp_master.sh
    ├─ submits rg_gen_batch.sh (array)
    │     └─ runs: python -m source.data_generation
    ├─ afterok -> submits rg_hist_manager.sh
    │     ├─ runs: python -m source.helpers
    │     └─ runs: python -m source.histogram_manager
    └─ repeats for RG steps 0..(steps-1)
```

- Taskfarm scripts: [run_rg.sh](../Taskfarm/scripts/run_rg.sh), [run_shifts.sh](../Taskfarm/scripts/run_shifts.sh), [rg_fp_master.sh](../Taskfarm/scripts/rg_fp_master.sh), [shifted_rg.sh](../Taskfarm/scripts/shifted_rg.sh), [rg_gen_batch.sh](../Taskfarm/scripts/rg_gen_batch.sh), [rg_hist_manager.sh](../Taskfarm/scripts/rg_hist_manager.sh), [gen_shifted_data.sh](../Taskfarm/scripts/gen_shifted_data.sh)
- Local helper: [Local/run_local.py](../Local/run_local.py) — a local driver that mirrors the per-task flow for development and small runs

1) High-level flow for an FP (fixed-point) run

- Submission: the Taskfarm helper scripts submit Slurm jobs using `sbatch`. For example, [run_rg.sh](../Taskfarm/scripts/run_rg.sh) calls `sbatch` to submit `rg_fp_master.sh`, and [run_shifts.sh](../Taskfarm/scripts/run_shifts.sh) submits `shifted_rg.sh`.
- Per-task work: code that generates matrix / t' samples and writes per-task histogram outputs (see [source/data_generation.py](../source/data_generation.py) and [source/histogram_manager.py](../source/histogram_manager.py)).
- Aggregation / launder: histograms are collected and either laundered (resampled) or symmetrised before being passed to the next RG step. The resample/launder behavior is implemented by `launder` in [source/utilities.py](../source/utilities.py) and the histogram build/save is performed in [Local/run_local.py](../Local/run_local.py) via `build_hist` + `save_data`.
- Iteration: the RG driver loops for `rg_settings.steps` iterations. The FP driver in the local helper is `rg_fp()` in [Local/run_local.py](../Local/run_local.py). On the cluster this is driven by Taskfarm orchestration scripts (<REMOTE_ROOT>/scripts).

2) High-level flow for an EXP / shifted run

- An EXP (exponent/shifted) run uses a fixed-point distribution as the starting point and applies a set of shifts. The local driver is `rg_exp()` in [Local/run_local.py](../Local/run_local.py) which loads a saved FP distribution, applies `shifts` from the config, and then runs RG steps for each shift.
- Producer: the FP distribution loaded by `rg_exp()` is read from an NPZ (path assembled in `Local/run_local.py`: `fp_data_file = f"{base_output_dir}/FP/hist/z/z_sym_hist_RG{rg_config.steps - 1}.npz"`).

3) Lifecycle of a single RG step (concrete mapping)

- Generation: a set of `t'` samples are produced by the RG mapping function at runtime. The mapping is invoked via `rg_data_workflow()` from [source/utilities.py](../source/utilities.py), called inside `rg_fp()` and `rg_exp()` in [Local/run_local.py](../Local/run_local.py).
- Histogram aggregation: the driver converts the sample array into histograms using `build_hist()` (see [Local/run_local.py](../Local/run_local.py)), which uses `numpy.histogram` and `get_density()` (`source/utilities.py`). Hist and edges are stored (NPZ files) via `save_data()`.
- Resampling / hand-off: depending on `engine.symmetrise` config, the code either centers the z-distribution (`center_z_distribution()` in [source/utilities.py](../source/utilities.py)) and then launder samples via `launder()` or directly launders the t-histogram. The resampled sample array becomes the input for the next RG step.

4) Where configs come from and how they are passed

- Canonical config templates live in `<REMOTE_ROOT>/scripts/configs/iqhe.yaml` and `Local/configs/local_iqhe.yaml`.
- CLI parsing and validation: [source/parse_config.py](../source/parse_config.py) exposes `build_parser()` and `validate_input()` which are used by [Local/run_local.py](../Local/run_local.py) (and likely by job submission wrappers) to construct runtime config dictionaries.
- The application-level config handler is `handle_config()` and `build_config()` in [source/config.py](../source/config.py); these functions are used by [Local/run_local.py](../Local/run_local.py) to create an `RGConfig` data structure consumed by `rg_fp()` / `rg_exp()`.
- `Local/run_local.py` writes a copy of the used/updated config via `save_updated_config(output_dir, config)` to the run output directory.

5) Artifacts written during the pipeline (high level)

- Per-step histograms: per-RG-step NPZ files (t and z histograms) — written by the RG drivers (see `save_data` calls in [Local/run_local.py](../Local/run_local.py)).
- Run manifest: `output_locs.json` — written at the end of `Local/run_local.py` and used to list produced histogram file locations.
- Config snapshot: the chosen config is written into the run output directory via `save_updated_config()` (see [source/config.py](../source/config.py)).

6) Failure modes & how to recover (practical)

- Missing output directories: drivers create directories with `Path(...).mkdir(parents=True, exist_ok=True)` in [Local/run_local.py](../Local/run_local.py). If a job fails before directories are created, re-create the expected tree and re-submit the job. Inspect the script that failed (Taskfarm logs / `output.txt` / `error.txt`).
-- Partial step outputs (e.g., missing NPZ for RGk): check for `t_hist_RG{k}.npz` / `z_hist_RG{k}.npz` in the run folder (see `Local/run_local.py` naming convention). If only some tasks wrote their tiles, re-run the aggregation stage for that RG step. The cluster master `<REMOTE_ROOT>/scripts/rg_fp_master.sh` submits per-step generation jobs (`rg_gen_batch.sh`) and histogram aggregation jobs (`rg_hist_manager.sh`) using `sbatch --dependency` to enforce ordering.
- Out of memory / OOM during sample generation: symptoms are abrupt job termination and large `samples` in config (see `rg_settings.samples` in `<REMOTE_ROOT>/scripts/configs/iqhe.yaml`). Reduce `rg_settings.samples` and/or `rg_settings.matrix_batch_size` and re-run locally first.

7) Where to look in code (mapping)

- CLI parsing & validation: [source/parse_config.py](../source/parse_config.py) — `build_parser()`, `validate_input()`
- Config handling: [source/config.py](../source/config.py) — `handle_config()`, `build_config()`, `save_updated_config()`
- Core RG drivers (../Local helper): [Local/run_local.py](../Local/run_local.py) — `rg_fp()`, `rg_exp()`
- Sample generation + mapping: [source/data_generation.py](../source/data_generation.py) (producer-side code; TODO: confirm which Taskfarm scripts invoke these functions)
- Histogram management: [source/histogram_manager.py](../source/histogram_manager.py)
- Utilities used across flow: [source/utilities.py](../source/utilities.py) — `rg_data_workflow()`, `launder()`, `center_z_distribution()`, `save_data()`, `get_density()`

8) Pipeline costs

- Generation job (rg_gen_batch.sh): CPU-heavy and memory-heavy - RAM usage for numpy array or matrix operations is the primary bottleneck
- Histogram job (rg_hist_manager.sh): IO-heavy - many file reads and writes on the shared FileSystem for histogram aggregation
See also: `Taskfarm/scripts/*` for cluster orchestration and `Local/run_local.py` for local runs.

Example Slurm invocation (short)

From the repository root the helper wrappers prepare an updated config then submit the Slurm master job. Recommended entrypoints:

```bash
# Submit an FP pipeline using the Taskfarm helper to a slurm system (recommended)
bash <REMOTE_ROOT>/scripts/run_rg.sh --config <REMOTE_ROOT>/scripts/configs/iqhe.yaml --set "rg_settings.steps=9" --set "engine.method=numerical" --out /path/to/updated_config_output
```

```bash
# Submit an EXP/shifted run for a particular shift index
bash <REMOTE_ROOT>/scripts/run_shifts.sh --config <REMOTE_ROOT>/scripts/configs/iqhe.yaml --index 0 --set "engine.method=numerical" --out /path/to/updated_config_output
```

Notes:
- Taskfarm submission: `run_rg.sh` and `run_shifts.sh` call `sbatch` to submit master job scripts (`rg_fp_master.sh`, `shifted_rg.sh`). The master job script then submits per-step jobs (`rg_gen_batch.sh`, `rg_hist_manager.sh`) with Slurm dependencies.
- NPZ layout: `source/utilities.py::save_data()` writes compressed NPZ archives with keys `histval`, `binedges`, and `bincenters` (via `np.savez_compressed(..., histval=..., binedges=..., bincenters=...)`).

Direct master sbatch (optional)

If you already have an updated config file (for example produced by `python source/parse_config.py` or by `run_rg.sh` with `--out`), you can submit the master Slurm job directly:

```bash
# Submit the master RG job directly with an updated config
sbatch <REMOTE_ROOT>/scripts/rg_fp_master.sh /path/to/updated_config.yaml
```

`rg_fp_master.sh` will then submit the per-step generation and histogram jobs (`rg_gen_batch.sh`, `rg_hist_manager.sh`) with Slurm dependencies as described above.
