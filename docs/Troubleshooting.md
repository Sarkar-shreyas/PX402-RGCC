# Troubleshooting

Common failure modes, symptoms, likely causes, and fixes.  For recovery procedures
see [Runbook.md](Runbook.md).

---

## 1. Missing output directories or `FileNotFoundError` on startup

**Symptoms:** job fails immediately with `FileNotFoundError` or `PermissionError`; no
`updated_config.yaml` in the expected folder.

**Likely cause:** `main.output_folder` in the config points to a non-existent or
unwritable path, or a required `.env` variable (`DATA_DIR`, `LOCAL_DIR`, etc.) is
missing.

**Fix:**
- Verify `.env` is present and all required variables are set (see
  [Config.md](Config.md)).
- `constants.py` raises `RuntimeError` at import time if any required variable is
  absent — check the traceback for the variable name.
- Create the target directory manually if needed; all drivers use
  `os.makedirs(..., exist_ok=True)`.

---

## 2. Partial step outputs (some NPZ files missing)

**Symptoms:** some `t_hist_RG{k}.npz` or `z_hist_RG{k}.npz` files exist but others
are absent; histogram aggregation fails or produces incomplete results.

**Likely cause:** one or more generation array tasks crashed or were preempted; network
or storage issues during the rsync step in `rg_gen_batch.sh`.

**Fix:**
- Check Slurm logs in `<REMOTE_ROOT>/job_logs/` for the failed task IDs.
- Confirm which `batch_*/READY` markers are present in the job outputs directory.
- Re-submit the generation array for the missing task indices, then re-run the
  histogram manager (`rg_hist_manager.sh`) with the same config and step index.

---

## 3. Out-of-memory errors during sample generation

**Symptoms:** job is killed abruptly by the OS; kernel OOM messages in the job log;
no output files for the affected step.

**Likely cause:** `rg_settings.samples` is too large for the available RAM, or
`rg_settings.matrix_batch_size` exceeds per-task memory limits.

**Fix:**
- Reduce `rg_settings.matrix_batch_size` in the config (controls the size of the
  matrix allocated per batch).
- Test locally with a small sample count before resubmitting.
- For very large runs, confirm the per-task memory request in the `#SBATCH --mem`
  directive in `rg_gen_batch.sh` is sufficient.

---

## 4. `KeyError` when loading an NPZ file in `rg_exp()`

**Symptoms:** `KeyError: 'histval'` (or `'binedges'`, `'bincenters'`) when the EXP
driver attempts to load the FP distribution.

**Likely cause:** the NPZ file was written with different key names, or the file is
corrupted.

**Fix:**
```python
import numpy as np
data = np.load("path/to/file.npz", allow_pickle=False)
print(list(data.keys()))   # expected: ['histval', 'binedges', 'bincenters']
```
If keys are wrong, the file must be regenerated.  All correct NPZ files in this
pipeline are written by `source/utilities.py::save_data` with these exact key names.

---

## 5. Config override (`--set`) not applied

**Symptoms:** the run uses the YAML default value despite a `--set` override being
passed on the command line.

**Likely cause:** shell quoting split the `key=value` argument, or the key path is
wrong.

**Fix:**
- Wrap each override in quotes: `--set "rg_settings.steps=3"`.
- Check that the key path uses dots to separate nested levels:
  `"parameter_settings.z.bins=500"`.
- If multiple overrides are passed, they can be chained after a single `--set`:
  `--set "rg_settings.steps=3" "rg_settings.samples=1000000"`.

---

## 6. YAML key validation error (`KeyError: Key X must be all lowercase`)

**Symptoms:** `build_config` or `handle_config` raises `KeyError` immediately after
loading a YAML file.

**Likely cause:** a config key contains an uppercase letter.

**Fix:** Convert all YAML keys to lowercase.  Values (e.g. `"FP"`, `"EXP"`) may be
uppercase; only the keys are restricted.

---

## 7. Symmetrisation produces an off-centre or asymmetric z-distribution

**Symptoms:** after enabling `engine.symmetrise = 1` the z-histogram peak is not
centred at zero; or the FP and EXP distributions look qualitatively different from
prior runs.

**Likely cause:** the `center_z_distribution` → `launder` branch in
`Local/run_local_iqhe.py` is being applied to a non-critical starting distribution, or
`engine.method` was changed between the FP and EXP runs.

**Fix:**
- Ensure the FP run converges (check `stats/*_moments.json` or the convergence output)
  before running EXP.
- Use the same `engine.method` for both FP and EXP runs.

---

## 8. QSHE aggregation fails with `RuntimeError: Block N incomplete`

**Symptoms:** `source/qshe_data_agg.py` exits with `RuntimeError: Block N incomplete`
before writing any aggregated files.

**Likely cause:** the generation job for block N did not complete (no `DONE` sentinel
in `{OUTPUT_DIR}/q{N}/`).

**Fix:**
- Check whether the generation job for that block exited successfully in its Slurm log.
- Re-run `python -m source.qshe_data_gen` for the missing block with the same
  arguments.
- Once the `DONE` file is present, re-run `source/qshe_data_agg.py`.

---

## 9. Notebook cell fails with `NameError` or `FileNotFoundError`

**Symptoms:** a notebook cell raises `NameError` (e.g. `name 'fits' is not defined`)
or `FileNotFoundError` when loading `p_data_agg.npy`.

**Likely cause:**
- The notebook was not run top-to-bottom; variables from earlier cells are missing.
- `dataversion` at line 977 of `test_qshe.py` does not match an existing directory
  under `{DATA_DIR}`.
- `.env` is not configured or `DATA_DIR` does not exist.

**Fix:**
- Use **Kernel → Restart and Run All** to execute cells in order.
- Update `dataversion` to match your actual data directory name.
- Verify `DATA_DIR` in `.env` and that the aggregated files exist:
  `{DATA_DIR}/{dataversion}/QP/data/p_data_agg.npy`.

---

## 10. `isdigit` check in `file_management.py` never triggers the error branch

**Symptoms:** passing an invalid `--step` argument to `file_management.py` does not
raise the expected `ValueError`.

**Cause (known bug):** line 285 of `file_management.py` reads
`elif str(args.step).isdigit:` — `isdigit` without `()` is always truthy (method
object reference rather than call).  The `raise ValueError` on the next line is
unreachable dead code.

**Workaround:** validate `--step` values manually before invoking the script.  This
is a low-priority cleanup target recorded in [CLAUDE.md](../CLAUDE.md).
