"""QSHE q-p parameter space scan — HPC data-generation stage.

Physical context
----------------
This module scans a two-dimensional (q, p) parameter grid for the Quantum Spin
Hall Effect (QSHE) model.  For each (q, p) pair it runs ``num_samples`` Monte
Carlo trials, each evolved through ``num_steps`` RG iterations.  The resulting
per-trial observables (selected by ``rg_config.outputs``) populate two arrays:

- **p_trial_data** — observable values after iterating the *p* RG map.
- **q_trial_data** — observable values after iterating the *q* RG map.

Both arrays have shape ``(len(qs), p_num, num_steps, met_dim)`` where
``met_dim`` is 3 when ``metric == "all"`` and 1 otherwise.

Pipeline role
-------------
This module is the **first stage** of the two-stage QSHE pipeline:

1. **Generation (HPC)** — ``source/qshe_data_gen.py`` (this file).
   Launched via Slurm on ``vulcan2``; one job array element per q-block.
   Outputs are written to a shared filesystem directory supplied via
   ``OUTPUT_DIR``.

2. **Analysis (local, Jupyter)** — ``test_qshe.ipynb``.
   All post-processing, visualisation, and ν extraction is performed in
   the notebook.  There is no CLI analysis equivalent.

Usage
-----
Invoked as a script with positional arguments::

    python -m source.qshe_data_gen \\
        NUM_SAMPLES NUM_STEPS Q_BLOCK Q_BLOCK_SIZE PHI_SEED GEN_SEED OUTPUT_DIR

Arguments:
    NUM_SAMPLES   (int): Total MC samples per (q, p) pair.
    NUM_STEPS     (int): Number of RG iterations per trial.
    Q_BLOCK       (int): Zero-based index of the q-block assigned to this job.
    Q_BLOCK_SIZE  (int): Number of q-values in each block.
    PHI_SEED      (int): NumPy PCG64 seed for random-phase generation.
    GEN_SEED      (int): NumPy PCG64 seed for trial generation.
    OUTPUT_DIR    (str): Shared filesystem directory for final outputs.

Output format
-------------
Two ``.npy`` files are written per q-block inside ``OUTPUT_DIR/q{Q_BLOCK}/``:

- ``p_data_q{Q_BLOCK}_{NUM_SAMPLES}_samples.npy`` — float64 array,
  shape ``(len(qs), p_num, num_steps, met_dim)``.
- ``q_data_q{Q_BLOCK}_{NUM_SAMPLES}_samples.npy`` — float64 array,
  same shape as above.
- ``DONE`` — empty sentinel file written on successful completion.

The q-axis is linearly spaced over ``[q_start * qsep, q_end * qsep]``
(derived from ``rg_config.q_min``, ``rg_config.q_num``).  The p-axis is
linearly spaced over ``[rg_config.p_min, rg_config.p_max]`` with
``rg_config.p_num`` points.

Note
----
Analysis of QSHE outputs is performed in ``test_qshe.ipynb`` — there is no
CLI analysis equivalent.
"""

import numpy as np
from pathlib import Path
import shutil
import sys
import os
from source.config import get_rg_config
from source.utilities import build_rng, qp_trials, generate_random_phases
from time import time

if __name__ == "__main__":
    # Load input params
    if len(sys.argv) == 8:
        num_samples = int(sys.argv[1].strip())
        num_steps = int(sys.argv[2].strip())
        q_block = int(sys.argv[3].strip())
        q_block_size = int(sys.argv[4].strip())
        phi_seed = int(sys.argv[5].strip())
        gen_seed = int(sys.argv[6].strip())
        output_dir = sys.argv[7].strip()
    else:
        raise SystemExit(
            "Usage: qshe_data_gen.py NUM_SAMPLES NUM_STEPS Q_BLOCK Q_BLOCK_SIZE PHI_SEED GEN_SEED OUTPUT_DIR"
        )

    # Load config and set up default params
    phi_rng = build_rng(phi_seed)
    gen_rng = build_rng(gen_seed)
    rg_config = get_rg_config()
    if rg_config.model != "qshe":
        raise SystemExit(f"Invalid model {rg_config.model}.")

    batch_size = rg_config.matrix_batch_size
    if batch_size > num_samples:
        batch_size = num_samples
    # p_vals = np.linspace(rg_config.p_min, rg_config.p_max, rg_config.p_num)
    if rg_config.metric == "all":
        met_dim = 3
    else:
        met_dim = 1

    # Setup temp and shared output folders
    temp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    local_block_dir = temp_dir / f"q{q_block}"
    shared_block_dir = Path(output_dir) / f"q{q_block}"

    # In case there's an old existing dir from failed runs
    if local_block_dir.exists():
        shutil.rmtree(local_block_dir)
    local_block_dir.mkdir(parents=True, exist_ok=False)

    if shared_block_dir.exists():
        raise RuntimeError(f"{shared_block_dir} folder already exists")

    print("-" * 100)
    print(f"Beginning q-p trials for q = {q_block}")
    start = time()

    # --- Build q and p grid axes for this block ---
    # q values are scaled so the full q-axis spans [0, 0.5] across all blocks
    q_start = rg_config.q_min + q_block * q_block_size
    q_end = min((q_block + 1) * q_block_size, rg_config.q_num)
    qsep = 0.5 / (rg_config.q_num - 1)
    qs = np.linspace(q_start * qsep, q_end * qsep, q_block_size)
    ps = np.linspace(rg_config.p_min, rg_config.p_max, rg_config.p_num)

    # Pre-allocate output arrays: axes are (q, p, step, metric_dim)
    p_trial_data = np.empty(
        shape=(len(qs), rg_config.p_num, num_steps, met_dim), dtype=np.float64
    )
    q_trial_data = np.empty(
        shape=(len(qs), rg_config.p_num, num_steps, met_dim), dtype=np.float64
    )

    # Generate a single shared phase array reused across all (q, p) pairs
    phis = generate_random_phases(num_samples, phi_rng, 16)
    assert len(qs) <= q_block_size

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
            if j[0] % 100 == 0:
                print(f"Trial no. {j[0]} completed in {time() - start:.3f} seconds.")

    # --- Save results to local scratch, then move atomically to shared FS ---
    p_filename = f"p_data_q{q_block}_{num_samples}_samples.npy"
    q_filename = f"q_data_q{q_block}_{num_samples}_samples.npy"
    np.save(local_block_dir / p_filename, p_trial_data)
    np.save(local_block_dir / q_filename, q_trial_data)

    # Add a flag to check if the job was successful
    (local_block_dir / "DONE").touch()

    # Move data from temp folder back to shared FS
    shutil.move(str(local_block_dir), str(shared_block_dir))

    print(
        f"q-p trial for q = {q_block} completed and saved to {output_dir} after {time() - start:.3f} seconds"
    )
    print("-" * 100)
