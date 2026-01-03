**Deployment / Staging model**

- The cluster does not run from a full git checkout. Runtime code is staged to `<REMOTE_ROOT>/code/source/` and scripts/configs to `<REMOTE_ROOT>/scripts/` by `file_management.py`.

When something fails on the cluster, follow these steps (assume you have SSH access to `<HOST>` and the runtime assets are staged under `<REMOTE_ROOT>`):

1) Inspect job logs and outputs

- Check job stdout/stderr files produced by Slurm under the working directories on the cluster. Master and worker scripts are `<REMOTE_ROOT>/scripts/rg_fp_master.sh`, `<REMOTE_ROOT>/scripts/rg_gen_batch.sh`, `<REMOTE_ROOT>/scripts/rg_hist_manager.sh`.
- Check the remote job outputs under `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/` for per-step artifacts and `READY` markers.

2) Recover or re-run a step

- If a generation batch failed: inspect the batch directory on the cluster and re-run the generation array for the missing batches using the staged scripts in `<REMOTE_ROOT>/scripts/`.
- If aggregation failed: re-run `<REMOTE_ROOT>/scripts/rg_hist_manager.sh` on the cluster with the same updated config YAML.

3) Pull outputs to local machine for inspection

- Use `file_management.py` to pull job outputs back to your local machine. Example (from `<LOCAL_REPO_ROOT>`):

```bash
python file_management.py --action pull --pull hist --version <version> --type FP --sys linux
```

Notes: use `file_management.py` as the authoritative tool for moving files between local and remote; it implements the exact remote folders and transfer method.

# Runbook (HPC-focused)

This runbook explains how to start, monitor and safely restart the pipeline. It treats the Taskfarm/Slurm path as authoritative and `Local/run_local.py` as a helper for testing.

Entrypoints (observed in repo):

- Taskfarm scripts: [Taskfarm/scripts/run_rg.sh](../Taskfarm/scripts/run_rg.sh), [Taskfarm/scripts/run_shifts.sh](../Taskfarm/scripts/run_shifts.sh), [Taskfarm/scripts/rg_fp_master.sh](../Taskfarm/scripts/rg_fp_master.sh)
- Local helper: [Local/run_local.py](../Local/run_local.py) — for quick local FP/EXP runs and debugging

Example execution (HPC):

The Taskfarm entry scripts submit Slurm jobs using `sbatch`. `Taskfarm/scripts/run_rg.sh` calls `sbatch` to submit `rg_fp_master.sh`, and `Taskfarm/scripts/run_shifts.sh` calls `sbatch` to submit `shifted_rg.sh` (see those scripts for the exact arguments passed).

- `--set` overrides maybe be written consecutively as shown in the 2nd example. Where multiple entries without a new command are found, they will be appended to `--set` input.

Example local invocation (proven in repo):

```bash
python -m Local.run_local --config Local/configs/local_iqhe --set "rg_settings.steps=2" --set "rg_settings.samples=10000" --type FP
python -m Local.run_local --config Local/configs/local_iqhe --set "rg_settings.steps=2" "rg_settings.samples=10000" --type FP --set "engine.method=numerical"
```

Where to find logs and outputs

- Local runs: `output.txt` and `error.txt` are created inside the run `output_dir` by [Local/run_local.py](../Local/run_local.py). The run also writes `output_locs.json` listing produced NPZ files.
Cluster runs: the master Slurm script `Taskfarm/scripts/rg_fp_master.sh` sets SBATCH directives and runtime log locations. Example directives in that script:

- `#SBATCH --output=../job_outputs/bootstrap/%x_%A.out`
- `#SBATCH --error=../job_logs/bootstrap/%x_%A.err`

At runtime the master script creates per-run folders and redirects stdout/stderr into per-run files under `job_outputs/${VERSIONSTR}/$TYPE` and `job_logs/${VERSIONSTR}/$TYPE` (see variables `joboutdir` and `logsdir` inside `rg_fp_master.sh`). Per-step generation and histogram jobs are also submitted via `sbatch` with `--output`/`--error` patterns (see `rg_gen_batch.sh` / `rg_hist_manager.sh` invocations in `rg_fp_master.sh`).

Safe restart procedures

- Restart from scratch:
  - Remove the target run folder (or move it aside). Confirm that no other jobs are writing to the same folder.
  - Re-submit the Taskfarm entry script (see Taskfarm/scripts).

- Restart from RG step k (partial restart):
  - Identify the last successfully written histograms: look for `t_hist_RG{k}.npz` / `z_sym_hist_RG{k}.npz` (naming from [Local/run_local.py](../Local/run_local.py)).
  - If all tiles for RG{k} exist and downstream RG{k+1} has not run, re-run the aggregation/driver for RG{k+1} only. The master script `Taskfarm/scripts/rg_fp_master.sh` enqueues `rg_gen_batch.sh` (generation) and `rg_hist_manager.sh` (aggregation) with Slurm dependencies; re-submitting the appropriate aggregation job or running `rg_hist_manager.sh` with the same config and step index will reproduce the aggregation stage for that RG step.
  - If RG{k} is partially complete (missing some tile NPZs), re-run the generator jobs for the missing tiles and then re-run aggregation.

Handling partial failures

- Generator jobs crashed (missing per-task NPZs): re-run the generator tasks for the missing task IDs. The per-task code is likely in `source/data_generation.py`; check Taskfarm submission arguments for task IDs (see Taskfarm/scripts).
- Aggregation/hist jobs crashed: confirm the presence of per-task hist files and re-run the aggregation step. The local aggregation logic is visible in [Local/run_local.py](../Local/run_local.py) and [source/histogram_manager.py](../source/histogram_manager.py).

Safety notes / assumptions

- Large `rg_settings.samples` and `parameter_settings.z.bins` can cause high memory and disk usage. Adjust `rg_settings.matrix_batch_size` to control memory footprint (config key exists: see `Taskfarm/configs/iqhe.yaml`).
- Always keep a copy of the FP NPZ corresponding to the final FP (used by EXP). The `rg_exp()` driver expects a saved FP NPZ at a particular path (see `Local/run_local.py`).

Cluster job details & example sbatch invocations

The Taskfarm scripts orchestrate submission of a master job which in turn submits per-step jobs. Concrete examples and behavior (extracted from the scripts):

- Entry submission (helper): `Taskfarm/scripts/run_rg.sh` prepares an updated config and calls:

```bash
sbatch --parsable "$scriptsdir/rg_fp_master.sh" "$UPDATED_CONFIG"
```

- Master job (`rg_fp_master.sh`) then submits per-step generation jobs. Example submission pattern used by the master script:

```bash
sbatch --parsable \
  --output=../job_outputs/bootstrap/rg_gen_RG${step}_%A_%a.out \
  --error=../job_logs/bootstrap/rg_gen_RG${step}_%A_%a.err \
  "$scriptsdir/rg_gen_batch.sh" \
    "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$METHOD" "$EXPR" "$INITIAL" "$EXISTING_T"
```

- The generation job script (`rg_gen_batch.sh`) is submitted as an array job (see SBATCH header `--array=0-31%4` inside the script). It expects the following positional args when invoked:

  1. `UPDATED_CONFIG`
  2. `TYPE` (FP/EXP)
  3. `VERSION` / `VERSIONSTR`
  4. `N` (total samples)
  5. `RG_STEP` (step index)
  6. `SEED`
  7. `METHOD` (analytic/numerical)
  8. `EXPR` (expression name)
  9. `INITIAL` (0/1)
 10. `EXISTING_T_FILE` (optional path)
 11. `SHIFT` (optional for EXP)

- Important behaviors in `rg_gen_batch.sh` (observed):
  - Uses `SLURM_ARRAY_TASK_ID` to compute `TASK_ID` and derive a deterministic `JOB_SEED` per task.
  - Writes per-batch files into a temporary directory `${TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}` and rsyncs back to a shared `batchdir` (with a `READY` marker file on success).
  - Calls `python -m source.data_generation` with arguments: `BATCH_SIZE`, `batchsubdir`, `INITIAL`, `RG_STEP`, `JOB_SEED`, and optionally `T_INPUT`.

- The master script submits a histogram manager job after generation, for example:
- Note that: $scriptsdir = <REMOTE_ROOT>/scripts
```bash
sbatch --parsable \
  --dependency=afterok:${gen_job} \
  --output=../job_outputs/bootstrap/rg_hist_RG${step}_%A.out \
  --error=../job_logs/bootstrap/rg_hist_RG${step}_%A.err \
  "$scriptsdir/rg_hist_manager.sh" \
    "$UPDATED_CONFIG" "$TYPE" "$VERSIONSTR" "$N" "$step" "$SEED" "$SYMMETRISE"
```

- `rg_hist_manager.sh` behaviors and expectations:
  - Validates presence of `batch_*` directories and a `READY` marker in each.
  - Iterates over batches and calls `python -m "source.helpers"` and `python -m "source.histogram_manager"` to convert t'→z, and to append/construct aggregate histograms.
  - Writes temporary histograms locally (under `TMPDIR`) and moves them to shared `jobdatadir` once complete.
  - Performs runtime assertions using Python to ensure NPZ files contain keys `histval` and `binedges` before continuing.
  - Produces additional artifacts: laundered t batches (`t_laundered_RG{step}_batch_{batch}.npy`), `input_t_hist_RG{step}.npz`, symmetrised histogram `sym_z_hist_RG{step}.npz` (for FP), and `DONE.json` metadata per RG step.

Restarting a failed aggregation job

- If `rg_hist_manager.sh` fails due to missing batches, re-run only that histogram manager job with the same positional args after ensuring missing `batch_*` folders are populated.
- If some `batch_*` folders exist and others do not, re-run the generation array job (`rg_gen_batch.sh`) for the missing task indices (submit array with appropriate `--array` or individual `sbatch` calls for missing indices), then re-run `rg_hist_manager.sh`.
