"""QSHE trial aggregation — second stage of the HPC data-generation pipeline.

Purpose
-------
This module collects the per-q-block ``.npy`` files produced by
``source/qshe_data_gen.py`` (one Slurm array element per block) and
concatenates them along the q-axis into a single consolidated array per
observable variable.

Expected inputs
---------------
For each q-block index *b* in ``range(block_num)`` and each observable *var*
in ``VARS``, the module reads::

    OUTPUT_DIR/q{b}/{var}_data_q{b}_{NUM_SAMPLES}_samples.npy

Each file must have shape ``(q_block_size, p_num, num_steps, met_dim)`` and
must be accompanied by a ``OUTPUT_DIR/q{b}/DONE`` sentinel file.  If any
sentinel is absent the run raises :exc:`RuntimeError` immediately.

Output format
-------------
For each variable *var* a single aggregated file is written::

    OUTPUT_DIR/{var}_data_agg.npy  — float64, shape (q_num, p_num, num_steps, met_dim)

A JSON state file is also written::

    OUTPUT_DIR/trial_state.json

containing q/p grid metadata, observable names, and run-configuration
parameters (samples, steps, fixed, metric, block size).

Aggregation strategy
--------------------
Blocks are concatenated in order along axis 0 (the q-axis) by direct array
assignment — no averaging is performed.  The final block may be smaller than
``q_block_size`` when ``q_num`` is not an exact multiple; this remainder is
handled explicitly by back-filling the tail of the output array.

Pipeline role
-------------
This is the **second stage** of the two-stage QSHE HPC pipeline:

1. ``source/qshe_data_gen.py`` — per-block trial generation (Slurm array job).
2. ``source/qshe_data_agg.py`` (this file) — aggregation into a single dataset.

Subsequent analysis is performed in ``test_qshe.ipynb``.

Usage
-----
Invoked as a script::

    python -m source.qshe_data_agg \\
        NUM_SAMPLES NUM_STEPS Q_BLOCK_SIZE OUTPUT_DIR VARS

Arguments:
    NUM_SAMPLES    (int): Total MC samples per (q, p) pair used during generation.
    NUM_STEPS      (int): Number of RG iterations per trial.
    Q_BLOCK_SIZE   (int): Number of q-values per block (must match generation).
    OUTPUT_DIR     (str): Directory containing the per-block ``q{b}/`` sub-folders.
    VARS           (str): Comma-separated observable names to aggregate
                          (e.g. ``"p,q"``).
"""

import math
from pathlib import Path
import numpy as np
import sys
from source.config import QSHEConfig, get_rg_config
import json
from time import time


def store_state(
    config: QSHEConfig, output_dir: str | Path, vars: list, block_size: int
):
    """Serialise run metadata to ``trial_state.json`` in the output directory.

    Writes a JSON file that records the q/p grid parameters, observable names,
    and key run-configuration values so that downstream analysis (e.g. in
    ``test_qshe.ipynb``) can reconstruct the grid without re-reading the config.

    Args:
        config: Validated :class:`~source.config.QSHEConfig` for the run.
            Used to extract q/p grid bounds, step counts, and aggregation
            settings.
        output_dir: Directory in which to write ``trial_state.json``.
            The file is created (or overwritten) at
            ``{output_dir}/trial_state.json``.
        vars: List of observable variable names that were aggregated
            (e.g. ``["p", "q"]``).  Stored verbatim in the JSON under the
            ``"vars"`` key.
        block_size: Effective q-block size used during aggregation.  For the
            final (potentially smaller) block this reflects the actual number
            of q-values in that block rather than the nominal block size.

    Side effects:
        Writes ``trial_state.json`` to *output_dir* and prints its path.

    Output schema::

        {
          "q":      {"Min": float, "Max": float, "Num": int, "Step": float},
          "p":      {"Min": float, "Max": float, "Num": int, "Step": float},
          "vars":   [str, ...],
          "config": {"Samples": int, "Steps": int, "Fixed": bool,
                     "Metric": str, "Block size": int}
        }
    """
    n = config.samples
    k = config.steps
    qstep = (config.q_max - config.q_min) / config.q_num
    pstep = (config.p_max - config.p_min) / config.p_num
    fix = config.fixed
    met = config.metric
    state = {
        "q": {
            "Min": config.q_min,
            "Max": config.q_max,
            "Num": config.q_num,
            "Step": qstep,
        },
        "p": {
            "Min": config.p_min,
            "Max": config.p_max,
            "Num": config.p_num,
            "Step": pstep,
        },
        "vars": vars,
        "config": {
            "Samples": n,
            "Steps": k,
            "Fixed": bool(fix),
            "Metric": met,
            "Block size": block_size,
        },
    }
    state_filename = f"{output_dir}/trial_state.json"
    with open(state_filename, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State config saved to {state_filename}")


if __name__ == "__main__":
    # Load input params
    if len(sys.argv) == 6:
        num_samples = int(sys.argv[1].strip())
        num_steps = int(sys.argv[2].strip())
        q_block_size = int(sys.argv[3].strip())
        output_dir = sys.argv[4].strip()
        vars = [v.strip() for v in sys.argv[5].split(",")]
    else:
        raise SystemExit(
            "Usage: qshe_data_agg.py NUM_SAMPLES NUM_STEPS Q_BLOCK_SIZE OUTPUT_DIR VARS"
        )

    # --- Load config and derive aggregation dimensions ---
    rg_config = get_rg_config()
    if rg_config.model != "qshe":
        raise SystemExit(f"Invalid model {rg_config.model}.")
    # metric=="all" produces 3 observables per trial; all other metrics produce 1
    if rg_config.metric == "all":
        met_dim = 3
    else:
        met_dim = 1
    print("Beginning aggregation step")
    print("-" * 100)
    start = time()

    # --- Compute block layout: how many blocks and the size of the final partial block ---
    # ceil division ensures the final partial block is included
    block_num = math.ceil(rg_config.q_num / q_block_size)
    print(f"Num blocks = {block_num}")
    # number of q-values in the last block (may be smaller than q_block_size)
    final_block_size = (block_num * q_block_size) - rg_config.q_num
    output_path = Path(output_dir)

    # --- Aggregate each observable variable across all q-blocks ---
    for var in vars:
        # consolidated output: one file per observable across all q-blocks
        agg_filename = output_path / f"{var}_data_agg.npy"
        final_data = np.empty(
            shape=(rg_config.q_num, rg_config.p_num, num_steps, met_dim),
            dtype=np.float64,
        )
        print(f"Final data shape = {final_data.shape}")
        for block in range(block_num):
            # each block lives in its own sub-directory: OUTPUT_DIR/q{block}/
            block_path = output_path / f"q{block}"
            # sentinel file written by qshe_data_gen.py on successful completion
            if not (block_path / "DONE").exists():
                raise RuntimeError(f"Block {block} incomplete.")
            # filename pattern mirrors what qshe_data_gen.py writes
            trial_data_file = (
                f"{output_dir}/q{block}/{var}_data_q{block}_{num_samples}_samples.npy"
            )
            trial_data = np.load(trial_data_file)
            if block == block_num - 1:
                # back-fill the tail when q_num is not a multiple of q_block_size
                q_block_size = final_block_size
                print(f"Final block {block} has size = {q_block_size}")
                start_index = rg_config.q_num - final_block_size
                end_index = rg_config.q_num
                final_data[start_index:end_index, :, :, :] = trial_data
            else:
                # standard block: contiguous slice along the q-axis
                start_index = block * q_block_size
                end_index = min((block + 1) * q_block_size, rg_config.q_num)
                final_data[start_index:end_index, :, :, :] = trial_data
            if block % 5 == 0:
                print(f"Agg done for block {block}.")
        np.save(agg_filename, final_data)
        print(f"Trial data for {var} aggregated after {time() - start:.3f} seconds")
        print(f"Agg data saved to {agg_filename}")
    print("-" * 100)
    # --- Serialise grid metadata and run parameters for downstream analysis ---
    store_state(rg_config, output_dir, vars, q_block_size)
    print("-" * 100)
    print(
        f"All aggregation steps completed for {block_num} q blocks after {time() - start:.3f} seconds"
    )
