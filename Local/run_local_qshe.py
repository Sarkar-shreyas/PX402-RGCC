"""Local QSHE (q, p) parameter-grid driver for development and testing.

Purpose
-------
Single-process driver that replicates the Taskfarm QSHE pipeline
(``source/qshe_data_gen.py`` → ``source/qshe_data_agg.py``) at local scale.
For each point on a (q, p) grid it runs ``num_samples`` Monte Carlo trials,
each evolved through ``num_steps`` RG iterations, and writes the results in
a layout that ``test_qshe.ipynb`` can read directly.

Unlike the old histogram-based FP/EXP workflow (now removed), this driver
performs no symmetrisation, laundering, or histogram construction.  It is
the correct entry point for finite-size scaling analysis of the QSHE phase
diagram.

Differences from the Taskfarm scripts
--------------------------------------
- **No Slurm**: the entire grid is processed sequentially in one Python
  process.  The Taskfarm splits the q-axis into blocks across a job array.
- **No aggregation step**: the Taskfarm writes per-q-block ``.npy`` files
  later assembled by ``source/qshe_data_agg.py``; this driver writes the
  fully aggregated output directly.
- **Shared phase array**: a single random-phase array is generated once
  (using ``rg_config.seed``) and reused across all (q, p) pairs, matching
  the behaviour of ``source/qshe_data_gen.py``.

CLI Usage
---------
Run from the repository root::

    python -m Local.run_local_qshe \\
        --config Local/configs/local_qshe_qp

    python -m Local.run_local_qshe \\
        --config Local/configs/local_qshe_qp \\
        --set "rg_settings.samples=10000" "rg_settings.steps=5" \\
              "parameter_settings.q.num=10" "parameter_settings.p.num=10"

Output location
---------------
Outputs are written under ``Local data/`` in the repository root::

    Local data/{version}_{method}_{expr}/QP/
        config/updated_config.yaml
        data/p_data_agg.npy          — float64, shape (q_num, p_num, steps, met_dim)
        data/q_data_agg.npy          — float64, shape (q_num, p_num, steps, met_dim)
        data/trial_state.json        — grid metadata consumed by test_qshe.ipynb
        output.txt
        error.txt

To read these outputs in ``test_qshe.ipynb``, set ``DATA_DIR`` in ``.env``
to ``<repo root>/Local data/`` and set ``dataversion`` to
``{version}_{method}_{expr}`` (e.g. ``"rg_qshe_numerical_shreyas"``).

Notes
-----
- ``met_dim`` is 3 when ``rg_settings.metric = "all"`` (mean, median, std)
  and 1 otherwise.
- stdout and stderr are redirected to ``output.txt`` and ``error.txt``
  inside the QP output directory when the module is run as a script.
- Use a small sample count (e.g. ``--set "rg_settings.samples=10000"``) and
  a coarse grid for quick smoke-tests before running a full local sweep.
"""

import sys
import numpy as np
from pathlib import Path
from time import time

from source.parse_config import build_parser, get_project_root, validate_input
from source.config import (
    handle_config,
    save_updated_config,
    get_nested_data,
    build_config,
)
from source.utilities import (
    build_rng,
    generate_random_phases,
    qp_trials,
    get_current_date,
)
from source.qshe_data_agg import store_state


# ---------- Helper ---------- #


def build_default_output_dir(config: dict) -> Path:
    """Build the default local output root for a QSHE QP config.

    Constructs the path as::

        <repo_root>/Local data/{version}_{method}_{expr}

    The ``QP/`` subdirectory and its children are created by the caller.

    Args:
        config: Raw config dictionary (after overrides have been applied).
            Must contain ``main.version``, ``engine.method``, and
            ``engine.expr``.

    Returns:
        Absolute :class:`~pathlib.Path` for the run output root.  The
        directory is not created by this function.
    """
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    version_str = f"{version}_{method}_{expr}"
    root = get_project_root(1)
    return root / "Local data" / version_str


if __name__ == "__main__":
    start_time = time()
    cur_date = get_current_date()
    print(f"[{cur_date}]: Starting QSHE QP sweep.")

    # --- Build CLI argument parser and parse arguments ---
    parser = build_parser()
    args = parser.parse_args()

    # validate_input expects args.type; inject "QP" so the check passes
    # without adding a --type flag to the public interface.
    args.type = "QP"
    args_dict = validate_input(args)

    # --- Load YAML config, apply CLI overrides, build typed QSHEConfig ---
    config = handle_config(args_dict["config"], args.override)
    rg_config = build_config(config)

    # --- Resolve output directories ---
    if args.out is None:
        base_output_dir = build_default_output_dir(config)
    else:
        base_output_dir = Path(args.out)

    qp_dir = base_output_dir / "QP"
    config_dir = qp_dir / "config"
    data_dir = qp_dir / "data"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save the finalised config for reproducibility
    save_updated_config(config_dir, config)

    # --- Redirect stdout and stderr into per-run log files ---
    output_filename = str(qp_dir / "output.txt")
    error_filename = str(qp_dir / "error.txt")
    print(f"Redirecting stdout → {output_filename}")
    print(f"Redirecting stderr → {error_filename}")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    output_file = open(output_filename, "w")
    error_file = open(error_filename, "w")
    sys.stdout = output_file
    sys.stderr = error_file

    # --- Unpack run parameters ---
    num_samples = rg_config.samples
    num_steps = rg_config.steps
    batch_size = rg_config.matrix_batch_size
    if batch_size > num_samples:
        batch_size = num_samples

    # met_dim: 3 for "all" (mean, median, std), 1 for any single metric
    if rg_config.metric == "all":
        met_dim = 3
    else:
        met_dim = 1

    print(f"[{cur_date}]: QSHE QP sweep")
    print("-" * 80)
    print(f"  Config    : {args_dict['config']}")
    print(f"  Version   : {rg_config.version}_{rg_config.method}_{rg_config.expr}")
    print(f"  Samples   : {num_samples:,}")
    print(f"  Steps     : {num_steps}")
    print(f"  Metric    : {rg_config.metric}  (met_dim={met_dim})")
    print(f"  Fixed     : {bool(rg_config.fixed)}")
    print(f"  Seed      : {rg_config.seed}")
    print(f"  q grid    : [{rg_config.q_min}, {rg_config.q_max}] × {rg_config.q_num}")
    print(f"  p grid    : [{rg_config.p_min}, {rg_config.p_max}] × {rg_config.p_num}")
    print(f"  Batch size: {batch_size:,}")
    print("-" * 80)

    # --- Build (q, p) grid axes ---
    qs = np.linspace(rg_config.q_min, rg_config.q_max, rg_config.q_num)
    ps = np.linspace(rg_config.p_min, rg_config.p_max, rg_config.p_num)

    # --- Initialise RNGs: phi_rng for phase generation, gen_rng for trials ---
    # Two independent generators derived from the single config seed,
    # mirroring the PHI_SEED / GEN_SEED split used in qshe_data_gen.py.
    phi_rng = build_rng(rg_config.seed)
    gen_rng = build_rng(rg_config.seed + 1)

    # Generate a single shared phase array reused across all (q, p) pairs,
    # matching the behaviour of qshe_data_gen.py.
    phis = generate_random_phases(num_samples, phi_rng, 16)

    # --- Pre-allocate output arrays: axes are (q, p, step, metric_dim) ---
    p_trial_data = np.empty(
        shape=(rg_config.q_num, rg_config.p_num, num_steps, met_dim),
        dtype=np.float64,
    )
    q_trial_data = np.empty(
        shape=(rg_config.q_num, rg_config.p_num, num_steps, met_dim),
        dtype=np.float64,
    )

    print(f"Output array shape: {p_trial_data.shape}")
    sweep_start = time()

    # --- Main (q, p) trial loop: one qp_trials call per grid cell ---
    for i, q in np.ndenumerate(qs):
        for j, p in np.ndenumerate(ps):
            a, b = qp_trials(
                q,
                p,
                num_samples,
                num_steps,
                phis,
                gen_rng,
                rg_config.metric,
                rg_config.fixed,
                rg_config.outputs,
                rg_config.inputs,
                batch_size,
            )
            p_trial_data[i[0], j[0], :, :] = a
            q_trial_data[i[0], j[0], :, :] = b

        elapsed = time() - sweep_start
        print(
            f"q-slice {i[0] + 1}/{rg_config.q_num} (q={q:.4f}) done in {elapsed:.1f}s"
        )

    # --- Save aggregated arrays ---
    p_path = data_dir / "p_data_agg.npy"
    q_path = data_dir / "q_data_agg.npy"
    np.save(p_path, p_trial_data)
    np.save(q_path, q_trial_data)
    print(f"Saved p_data_agg.npy → {p_path}")
    print(f"Saved q_data_agg.npy → {q_path}")

    # --- Write trial_state.json for downstream analysis ---
    # block_size is a concept from the Taskfarm aggregation; for a local
    # single-process run there are no blocks, so q_num is used as a proxy.
    store_state(rg_config, data_dir, ["p", "q"], rg_config.q_num)

    end_time = time()
    print("-" * 80)
    print(
        f"QP sweep complete: {rg_config.q_num * rg_config.p_num} grid cells"
        f" in {end_time - sweep_start:.1f}s"
    )

    # --- Restore stdout/stderr and report completion ---
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    output_file.close()
    error_file.close()
    print(f"Outputs written to {qp_dir}")
    print(f"[{cur_date}]: Sweep completed after {end_time - start_time:.1f}s")
