Local testing notes

- `Local/run_local.py` is a developer convenience for single-process testing and should be run from your local repo (`<LOCAL_REPO_ROOT>`). It is not staged to the cluster and should not be assumed to exist on `<REMOTE_ROOT>`.

---

Quick local run example (from `<LOCAL_REPO_ROOT>`):

```bash
python -m Local.run_local --config Local/configs/local_iqhe --set "rg_settings.steps=2" --set "rg_settings.samples=10000" --type FP
```

What this does locally:
- Executes the same library code in `source/` as the staged runs, but on a single process and writing outputs to a local `output_dir`.
- Produces NPZ histograms and manifests in the local `output_dir` for inspection.

Notes about parity with the cluster:

- The staged cluster runtime uses the same `source/` library; ensure you push `source/` to `<REMOTE_ROOT>/code/source/` with `file_management.py` when moving to cluster testing.

# Local testing and quick sanity checks

Purpose: Enables RG runs with small sample sizes and step counts
| Local runs are not statistically equivalent to Taskfarm runs due to lack of array batching and aggregation

Local helper driver

- `Local/run_local.py` is the canonical local driver. It implements `rg_fp()` and `rg_exp()` and mirrors the pipeline loop used by the cluster.

Proven local invocation pattern (observed in repo):

```bash
python -m Local.run_local --config Local/configs/local_iqhe --set "rg_settings.steps=2" --set "rg_settings.samples=10000" --type FP
```

What is supported locally

- Full RG loop for FP and EXP modes via `rg_fp()` and `rg_exp()` in [Local/run_local.py](Local/run_local.py).
- Local building/saving of NPZ histogram artifacts using the same `save_data()` utility used by cluster code.

What is NOT supported (local vs cluster)

- The Taskfarm-style distributed generation and per-task job orchestration (Taskfarm/scripts) — local runs are single-process and do not reproduce job-array behavior.

Suggested tiny settings for quick checks

- `rg_settings.steps=2`
- `rg_settings.samples=10000` (or smaller for very fast runs)
- `parameter_settings.z.bins=200` (reduce histogram memory)

Ensuring local results match pipeline assumptions

- File layout and naming: local runs use the same NPZ naming scheme (`t_hist_RG{n}.npz`, `z_sym_hist_RG{n}.npz`) so downstream EXP drivers can load local FP outputs.
- RNG semantics: `Local/run_local.py` constructs an RNG from `rg_settings.seed` via `build_rng()` in [source/utilities.py](source/utilities.py). Use the same seed for reproducibility.

Logging and artifacts

- Local runs redirect stdout/stderr into `<output_dir>/output.txt` and `<output_dir>/error.txt` (see [Local/run_local.py](Local/run_local.py)). The final manifest `output_locs.json` is written to the run `output_dir`.
