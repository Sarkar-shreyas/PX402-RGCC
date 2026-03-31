# Runbook

Operational procedures for the most common tasks.  For pipeline internals see
[Pipeline.md](Pipeline.md); for config keys see [Config.md](Config.md).

---

## IQHE — full HPC production run

### 1. Prepare and push to the cluster

```bash
# Push source library, shell scripts, and config to the cluster
python file_management.py --action push --push code scripts config \
    --version fp_iqhe_numerical_shaw --sys linux
```

This stages:
- `source/` → `<REMOTE_ROOT>/code/source/`
- `Taskfarm/scripts/` → `<REMOTE_ROOT>/scripts/`
- `Taskfarm/configs/` → `<REMOTE_ROOT>/scripts/`

### 2. Submit the FP run (on the cluster)

```bash
bash Taskfarm/scripts/run_rg.sh \
    --config Taskfarm/configs/iqhe.yaml \
    --set "engine.method=numerical" \
    --out /tmp/configs
```

`run_rg.sh` validates the config, writes an updated YAML to `--out`, and submits
`rg_fp_master.sh` via `sbatch`.  The master script then submits generation array jobs
and histogram manager jobs with Slurm dependencies for each RG step.

### 3. Submit the EXP run (on the cluster, after FP completes)

```bash
bash Taskfarm/scripts/run_shifts.sh \
    --config Taskfarm/configs/iqhe.yaml \
    --index 0 --out /tmp/configs
```

`--index N` selects the Nth shift value from `data_settings.shifts`.  Run once per
shift index.

### 4. Pull histograms

```bash
# Pull FP histograms
python file_management.py --action pull --pull hist \
    --version fp_iqhe_numerical_shaw --type FP --sys linux

# Pull EXP histograms
python file_management.py --action pull --pull hist \
    --version fp_iqhe_numerical_shaw --type EXP --sys linux
```

### 5. Extract ν

```bash
python -m analysis.critical_exponent \
    --version fp_iqhe_numerical_shaw --mode EXP --steps 9
```

Outputs `peaks.json`, `overall_stats.json`, and PNG plots to the version folder.
See [Artifacts.md](Artifacts.md) for the full output layout.

---

## IQHE — local test run

```bash
# Minimal FP run (3 steps, 10M samples)
python -m Local.run_local_iqhe \
    --config Local/configs/local_iqhe \
    --set "rg_settings.steps=3" "rg_settings.samples=10000000" \
    --type FP

# Follow with EXP run using the same version
python -m Local.run_local_iqhe \
    --config Local/configs/local_iqhe \
    --set "rg_settings.steps=3" "rg_settings.samples=10000000" \
    --type EXP
```

Outputs land in `Local data/{version}/FP/` and `Local data/{version}/EXP/`.

For very fast sanity checks, reduce further:

```yaml
rg_settings.steps: 2
rg_settings.samples: 1000000
parameter_settings.z.bins: 500
```

---

## QSHE — HPC data generation

### 1. Push to the cluster

```bash
python file_management.py --action push --push code scripts config \
    --version qp_unfixed_numerical_shreyas --sys linux
```

### 2. Submit generation jobs (on the cluster)

The QSHE generation script is submitted as a Slurm array, one element per q-block:

```bash
sbatch Taskfarm/scripts/qshe_gen.sh \
    <NUM_SAMPLES> <NUM_STEPS> <Q_BLOCK_SIZE> <OUTPUT_DIR>
```

Each array element invokes `python -m source.qshe_data_gen` with positional arguments:

```
NUM_SAMPLES  NUM_STEPS  Q_BLOCK  Q_BLOCK_SIZE  PHI_SEED  GEN_SEED  OUTPUT_DIR
```

### 3. Aggregate (on the cluster, after all generation jobs complete)

```bash
python -m source.qshe_data_agg \
    NUM_SAMPLES NUM_STEPS Q_BLOCK_SIZE OUTPUT_DIR "p,q"
```

The aggregation script checks for `DONE` sentinels in every `q{b}/` sub-directory
before concatenating.  It writes `p_data_agg.npy`, `q_data_agg.npy`, and
`trial_state.json`.

### 4. Pull aggregated data

```bash
python file_management.py --action pull --pull data \
    --version qp_unfixed_numerical_shreyas --sys linux
```

### 5. Analyse in the notebook

Open `test_qshe.ipynb` and run all cells top-to-bottom.  The notebook expects:

```
{DATA_DIR}/{dataversion}/QP/data/p_data_agg.npy
{DATA_DIR}/{dataversion}/QP/data/q_data_agg.npy
{DATA_DIR}/{dataversion}/QP/config/updated_config.yaml
```

The `dataversion` variable is set at line 977 of `test_qshe.py`:

```python
dataversion = "qp_unfixed_numerical_shreyas"
```

Update this string to point to a different dataset.  See [QSHE.md](QSHE.md) for a
full walkthrough of the notebook.

---

## QSHE — local test run

```bash
# Minimal sweep (coarse grid, small sample count)
python -m Local.run_local_qshe \
    --config Local/configs/local_qshe_qp \
    --set "rg_settings.samples=10000" "rg_settings.steps=5" \
           "parameter_settings.q.num=10" "parameter_settings.p.num=10"

# Full sweep using the config defaults
python -m Local.run_local_qshe \
    --config Local/configs/local_qshe_qp
```

Outputs land in `Local data/{version}_{method}_{expr}/QP/`.  To analyse the result in
the notebook, set `DATA_DIR` in `.env` to `<repo root>/Local data/` and update
`dataversion` at line 977 of `test_qshe.py` to match.

---

## Monitoring cluster jobs

```bash
# Check job status
squeue -u $USER

# Inspect exit codes for a completed job
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed

# Check READY markers (indicates successful generation batch)
ls <REMOTE_ROOT>/job_outputs/<version>/FP/data/batch_*/READY
```

---

## Safe restart procedures

### Restart an IQHE run from scratch

1. Move or delete the existing output folder for the version (confirm no jobs are
   writing to it).
2. Re-submit via `run_rg.sh` as above.

### Restart from RG step k (partial restart)

1. Identify the last successfully written histograms:
   `<job_outputs>/{version}/FP/hist/t/t_hist_RG{k}.npz` and
   `<job_outputs>/{version}/FP/hist/sym_z/sym_z_hist_RG{k}.npz`.
2. If all batches for step k are present but step k+1 has not run, re-submit the
   histogram manager job with the same config.
3. If some batches for step k are missing, re-submit the generation array for the
   missing task indices only, then re-run the histogram manager.

### Re-run a failed aggregation

Confirm all `batch_*` directories contain a `READY` marker.  Re-run
`rg_hist_manager.sh` with the same positional arguments.

---

## Pushing changes after editing source code

```bash
# Push only code
python file_management.py --action push --push code \
    --version <version> --sys linux

# Push only scripts (e.g. after editing a .sh file)
python file_management.py --action push --push scripts \
    --version <version> --sys linux

# Push only config
python file_management.py --action push --push config \
    --version <version> --sys linux
```

Multiple targets can be combined:
`--push code scripts config`
