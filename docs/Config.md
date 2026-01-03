Deployment / Staging model

- Local config templates live in `Taskfarm/configs/` in the git repo and are uploaded by `file_management.py` to `<REMOTE_ROOT>/scripts/` on the cluster (see `file_management.py`).

Overrides and the staged config

- Wrapper scripts on the cluster expect the YAML config to be available under `<REMOTE_ROOT>/scripts/` when they run. If you edit a template locally, push it with `file_management.py --push config` before running the staged scripts.

Examples (push updated config templates to the cluster):

```bash
python file_management.py --push config --version <version> --sys linux
```

Notes

- `source/parse_config.py` is used by both local and staged runs for parsing and `--set` overrides — the same override keys apply in both environments.
- If in doubt about which config file is running on the cluster, inspect `<REMOTE_ROOT>/scripts/` on the cluster — that is the staged scripts/config location, as implemented in `file_management.py`.

# Configuration system

Primary config files (observed):

- `Taskfarm/configs/iqhe.yaml` — authoritative HPC run config template.
- `Local/configs/local_iqhe.yaml` — local testing config template used by `Local/run_local.py`.

Top-level sections (keys seen in example configs):

- `main`:
  - `version` (string)
  - `id` (string)
  - `type` (string) — e.g., `fp` or `exp` / `FP` / `EXP` variants seen in code usage
  - `output_folder` (string)
- `engine`:
  - `model` (string)
  - `method` (string) — `analytic` or `numerical` observed in code
  - `resample`
  - `expr` (string)
  - `symmetrise` (int/boolean flag)
- `rg_settings`:
  - `seed` (int)
  - `steps` (int)
  - `samples` (int)
  - `matrix_batch_size` (int)
- `data_settings`:
  - `inputs`, `outputs`, `shifts` (list)
- `parameter_settings`:
  - `z`:
    - `bins` (int)
    - `range` (2-element list)
  - `tprime`:
    - `bins` (int)
    - `range` (2-element list)
- `convergence`:
  - `msd_tol`, `std_tol`

Config invariants

- rg_settings.samples > matrix_batch_size
- parameter_settings.z.bins must be even
- engine.expr must match a supported expression `[jack, shaw, cain, test]`

How configs are used at runtime

- CLI building and validation: [source/parse_config.py](../source/parse_config.py) exposes `build_parser()` and `validate_input()` and is used by `Local/run_local.py`.
- The runtime config object is constructed via `handle_config()` and `build_config()` in [source/config.py](../source/config.py), which produce an `RGConfig` (as used in [Local/run_local.py](../Local/run_local.py)).
- `Local/run_local.py` writes a copy of the resolved config with `save_updated_config(output_dir, config)`.

Overrides and CLI usage (observed pattern)

- `Local/run_local.py` builds a parser and accepts `--config` and `--set` overrides. A sample invocation seen in the workspace:

```bash
python -m Local.run_local --config "Local/configs/local_iqhe" --set "rg_settings.steps=4" "rg_settings.samples=1000000" --type FP
python -m Local.run_local --config "Local/configs/local_iqhe" --set "engine.method=numerical" --set "rg_settings.steps=4" --type
```
- `--set` overrides maybe be written consecutively as shown in the 2nd example. Where multiple entries without a new command are found, they will be appended to `--set` input.

Type pitfalls and gotchas

- Numeric lists in YAML: `shifts` are numeric lists in YAML (`[0.003, 0.005, ...]`) but the code sometimes treats shifts as strings (see `constants.SHIFTS` which contains string values). Be careful when matching types between code and config.
- Large integer values: `rg_settings.samples` can be very large in Taskfarm templates. Ensure your shell and tools can handle large integers and that you do not lose precision via floating-point conversions.
- Quoting & CLI: when using `--set` to override nested keys, quote the whole `key=value` pair to avoid shell splitting (as shown in the example above).

Minimal example config snippet (verified keys only)

```yaml
main:
  version: "rg_iqhe"
  id: "local"
  type: "fp"

engine:
  method: "analytic"
  expr: "shaw"
  symmetrise: 1

rg_settings:
  seed: 1234
  steps: 7
  samples: 32000000

data_settings:
  shifts: [0.003, 0.005]

parameter_settings:
  z:
    bins: 50000
    range: [-25.0, 25.0]
```
