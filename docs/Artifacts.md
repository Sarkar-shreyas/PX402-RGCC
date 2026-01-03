Deployment note

- Artifacts written by the pipeline on the cluster are stored under `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/...` (this mapping is implemented in `file_management.py`). Pull them with `file_management.py`.

Primary artifact formats

- NPZ histograms: `histval`, `binedges`, `bincenters` — produced by `source/utilities.py::save_data()`.
- Laundered t batches: produced by the histogram manager and staged for the next RG step under the job outputs area on the cluster.

Where to find artifacts (local vs remote)

- Local (for small tests): `Local/run_local.py` writes outputs into the `output_dir` you configure locally.
- Remote (cluster): aggregated artifacts and job outputs live under `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/...` on `<HOST>`; use `file_management.py` to retrieve them.
- Note: `output_dir` on the remote server will live under <REMOTE_ROOT>/job_outputs

# Artifacts catalog

This document lists the primary artifacts produced by the pipeline, who writes them, and brief notes about regeneration.

| Artifact path / pattern | Producer (script/module) | When created | Contents | Size considerations | Safe to delete / regen |
|---|---:|---|---|---|---|
| <output_dir>/hist/t/t_hist_RG{step}.npz | `rg_fp()` / `rg_exp()` in [Local/run_local.py](Local/run_local.py) (cluster drivers produce equivalent files) | Each RG step | NPZ containing t-histogram arrays (saved via `save_data()`) | Can be large when `t_bins` is large; contains dense arrays | Regenerable by re-running step; safe to delete but will require re-run |
| <output_dir>/hist/z/z_{sym}hist_RG{step}.npz | `rg_fp()` / `rg_exp()` in [Local/run_local.py](Local/run_local.py) | Each RG step | NPZ containing z-histogram arrays (`histval` / `binedges` / `bincenters` observed in code) | z hist bins often large (`parameter_settings.z.bins`) — watch disk | Regenerable by re-running step (need upstream inputs) |
| <output_dir>/output.txt, <output_dir>/error.txt | `Local/run_local.py` (stdout/stderr redirected) | Per run (local) | Console log and stderr for the run | Small text files but can grow if job is verbose | Safe to delete; useful for debugging |
| <output_dir>/output_locs.json | `Local/run_local.py` | At the end of a local run | JSON manifest mapping RG steps to NPZ file paths | Small | Regenerable by re-running the run |
| <output_dir>/config_snapshot.yaml or similar | `save_updated_config()` via [source/config.py](source/config.py) | At run start | Config used for the run (full resolved config) | Small | Safe to keep; recommended to archive |

Notes:
- Output files for taskfarm jobs can be found in the job_outputs folder. Logs produced by run_local.py will print to the 'Local data' folder
- NPZ layout: the code that consumes FP runs (`rg_exp()` in [Local/run_local.py](Local/run_local.py)) expects keys `histval`, `binedges`, and `bincenters` when loading saved FP distributions. These keys are written by `source/utilities.py::save_data()` using `np.savez_compressed(..., histval=..., binedges=..., bincenters=...)`.
- Shared filesystem vs tmpdirs: the code constructs output paths from the project root and writes directly into configured output folders (see `build_default_output_dir()` and `output_dir` creation in [Local/run_local.py](Local/run_local.py)). If jobs are run on a cluster with local scratch, ensure a subsequent copy to the shared data directory before aggregation.

Regeneration notes:

- Per-step histograms are regenerable by re-running the corresponding RG step(s). If the cluster pipeline stages are separated (e.g., generation vs aggregation), re-run only the missing stage if possible.
- If you must free space quickly: remove intermediate NPZ steps far from the final output, but keep the FP distribution used by EXP runs if you intend to re-run exponent calculations.
