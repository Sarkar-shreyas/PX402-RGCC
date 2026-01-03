Deployment-aware troubleshooting

- Since runtime assets on the cluster are staged under `<REMOTE_ROOT>/`, always inspect the staged scripts and job outputs there first. `file_management.py` is the canonical mapping tool — use it to pull outputs locally for deeper inspection.

Common checks

1) Missing batches

- Inspect `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/data/` on the cluster for batch directories and `READY` markers.
- If a batch is missing from job outputs, check the generation array job stdout/stderr from the Slurm logs produced by the staged scripts in `<REMOTE_ROOT>/scripts/`.

2) NPZ load errors

- Verify NPZ keys `histval`, `binedges`, `bincenters` (written by `source/utilities.py::save_data()`). Pull the NPZ to your workstation using `file_management.py` and inspect locally.

3) Script mis-match on cluster

- If a script on the cluster behaves unexpectedly, confirm you pushed the correct script/config via `file_management.py --push scripts --push config` and that the staged files in `<REMOTE_ROOT>/scripts/` match your local `Taskfarm/scripts/` and `Taskfarm/configs/`.

Useful commands (examples run from `<LOCAL_REPO_ROOT>`):

```bash
# Pull the latest hist artifacts locally
python file_management.py --action pull --pull hist --version <version> --type FP --sys linux

# Re-upload scripts/configs after local edits
python file_management.py --push scripts --push config --version <version> --sys linux
```

TODO: For cluster-specific permission or environment issues, inspect the cluster environment directly — see your cluster admin or the `file_management.py` comments for further guidance.

# Troubleshooting

Common failure modes, symptoms, likely causes, and where to inspect.

1) Missing output directories or permission errors

- Symptoms: job fails early with `FileNotFoundError` or `PermissionError`. No `output_locs.json` in expected folder.
- Likely cause: wrong `output_folder` in config or user lacks write permission to target path.
- Where to inspect: `Local/run_local.py` (look for output dir creation using `Path(...).mkdir(parents=True, exist_ok=True)`), environment `.env` variables (top-level `.env`) and Taskfarm submission wrapper.
- Fix: ensure `main.output_folder` points to an existing writable directory, or create it manually and re-run.

2) Partial outputs for RG step (some NPZ files present, others missing)

- Symptoms: some `t_hist_RG{k}.npz` or `z_hist_RG{k}.npz` files missing; aggregation stage fails.
- Likely cause: per-task generator jobs crashed or were preempted; network/storage issues during write.
- Where to inspect: Task job logs (Taskfarm job stdout/stderr), per-task output directories in the run folder.
- Fix: re-run only the missing generator tasks; if unsure, re-run the aggregation for RG{k} after ensuring all per-task outputs are present.

3) OOM / memory errors during generation

- Symptoms: job killed by OS; abrupt termination; kernel OOM messages.
- Likely cause: `rg_settings.samples` is too large for available RAM or per-task `matrix_batch_size` is too big.
- Where to inspect: `Taskfarm/configs/iqhe.yaml` and the config snapshot written by `save_updated_config()` in the run folder.
- Fix: reduce `rg_settings.samples` and/or `rg_settings.matrix_batch_size` and re-run small-scale locally first.

4) Incompatible NPZ layout or load errors in `rg_exp()`

- Symptoms: `KeyError` when loading FP NPZ inside `rg_exp()` (`fp_data['histval']` missing).
- Likely cause: `save_data()` produced different key names than expected, or NPZ was corrupted.
- Where to inspect: `source/utilities.py::save_data()` and the NPZ consumer in [Local/run_local.py](Local/run_local.py) (`rg_exp()` loads `histval`, `binedges`, `bincenters`).
- Fix: open the NPZ with `numpy.load` and confirm keys, or re-create the NPZ with the expected keys.

5) Config parsing / override not applied

- Symptoms: overrides passed with `--set` do not take effect.
- Likely cause: incorrect override syntax; quoting issues in shell.
- Where to inspect: `source/parse_config.py::validate_input()` and `Local/run_local.py` (parser usage).
- Fix: wrap key/value pairs in quotes and verify `validate_input()` implementation. Example pattern used in repo:

```bash
--set "rg_settings.steps=4"
```

6) Misc: unexpected behavior in symmetrisation / launder

- Symptoms: resulting distributions appear shifted or non-centred after `symmetrise` is enabled.
- Likely cause: code path under `if symmetrise == 1:` in `Local/run_local.py` applies `center_z_distribution()` then `launder()`; differences may arise between analytic vs numerical `method` leading to differing sample sizes or phase handling.
- Where to inspect: the `symmetrise` branch in [Local/run_local.py](Local/run_local.py) and the implementations of `center_z_distribution()` and `launder()` in [source/utilities.py](source/utilities.py).

If you reach a blocker

- Collect the run folder and logs (`output.txt`, `error.txt`, `output_locs.json`) and the config snapshot saved in the output directory, then open the relevant functions mentioned above.

7) If nothing makes sense

- Check config
- Check last completed RG step
- Check Slurm exit codes via
```bash
sacct -j `job_id` `format=`
```
- check READY markers
