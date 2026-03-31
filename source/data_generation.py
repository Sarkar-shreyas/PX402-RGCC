#!/usr/bin/env python
"""Batch data-generation script for RG steps.

Purpose
-------
Produce the next-step amplitude samples ``t'`` for a single RG iteration.
The script is intended to be executed directly from the command line or
from a Slurm job-array task.  It performs no post-processing beyond saving
the raw sample array, so multiple instances can run in parallel without
interference.

Two operating modes
-------------------
- **Fresh start** (``INITIAL=1``): draws ``ARRAY_SIZE`` amplitudes from a
  flat ``P(g) = 1`` distribution (``t = √g``) via
  ``generate_initial_t_distribution``.
- **Continuation** (``INITIAL=0``): loads an existing ``(N,)`` float64
  amplitude array from ``EXISTING_T_FILE`` produced by a prior RG step.

CLI usage
---------
Fresh start::

    python -m source.data_generation ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED

Continuation::

    python -m source.data_generation ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED EXISTING_T_FILE

Arguments
---------
ARRAY_SIZE : int
    Number of MC amplitude samples to generate.
OUTPUT_DIR : str
    Directory in which the output ``.npy`` file is written (created if
    absent).
INITIAL : int
    ``1`` to draw a fresh distribution; ``0`` to load from
    ``EXISTING_T_FILE``.
RG_STEP : int
    Current RG iteration index, used only for labelling the output file
    and log messages.
SEED : int
    Integer seed passed to ``numpy.random.default_rng`` (PCG64 bit
    generator) to make the MC sequence fully reproducible.  Use the same
    seed on re-runs to obtain identical samples.
EXISTING_T_FILE : str, optional
    Path to a ``.npy`` file containing amplitude samples from the
    preceding RG step.  Required when ``INITIAL=0``; ignored (and
    internally set to ``"None"``) when ``INITIAL=1``.  Allows multi-step
    pipelines to chain RG iterations without regenerating earlier data.

Output file naming
------------------
The output array is saved as::

    {OUTPUT_DIR}/t_data_RG{RG_STEP}_{ARRAY_SIZE}_samples.npy

The file contains a 1-D ``float64`` array of ``ARRAY_SIZE`` amplitude
values ``t' ∈ [0, 1]``.  Method (analytic/numerical) and expression
variant are read from the ``RG_CONFIG`` environment variable at runtime.
"""

import os
import sys
import numpy as np
from source.utilities import (
    generate_initial_t_distribution,
    generate_random_phases,
    extract_t_samples,
    rg_data_workflow,
    build_rng,
)
from source.config import get_rg_config

# ---------------------------------------------------------------------------
# CLI entry point — positional argument order:
#   1. ARRAY_SIZE      int   — number of MC amplitude samples to generate
#   2. OUTPUT_DIR      str   — directory for the output .npy file
#   3. INITIAL         int   — 1 = fresh distribution, 0 = load from file
#   4. RG_STEP         int   — current RG iteration index (labelling only)
#   5. SEED            int   — NumPy PCG64 RNG seed for reproducibility
#   6. EXISTING_T_FILE str   — (optional) path to prior-step .npy samples;
#                              required when INITIAL=0, omitted when INITIAL=1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load input params, checking if we're starting RG steps or continuing from an input sample
    if len(sys.argv) == 6:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        seed = int(sys.argv[5].strip())
        existing_t_file = "None"
    elif len(sys.argv) == 7:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        seed = int(sys.argv[5].strip())
        existing_t_file = sys.argv[6].strip()
    else:
        raise SystemExit(
            "Usage: data_generation.py ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED [EXISTING_T_FILE]"
        )

    print("-" * 100)
    print(f"Beginning data generation for RG step {rg_step}")
    rng = build_rng(seed)
    rg_config = get_rg_config()
    method = rg_config.method
    expr = rg_config.expr
    batch_size = rg_config.matrix_batch_size
    if initial == 1:
        t = generate_initial_t_distribution(array_size, rng)
        print("Generated initial t distribution")
    else:
        print(f"Using t data from {existing_t_file}")
        t = np.load(existing_t_file)
    if method.lower()[0] == "a":
        i = 4  # Analytic t' has 4 reduced loop phases
    elif method.lower()[0] == "n":
        i = 8  # A matrix has 8 unique phases
    else:
        raise ValueError(
            "Unsupported method selected. method: a = Analytic, n = Numerical"
        )
    phases = generate_random_phases(array_size, rng, i)
    t_array = extract_t_samples(t, array_size, rng)
    t_prime = rg_data_workflow(method, t_array, phases, array_size, expr, batch_size)
    t_filename = os.path.join(
        output_dir, f"t_data_RG{rg_step}_{array_size}_samples.npy"
    )
    os.makedirs(output_dir, exist_ok=True)
    np.save(t_filename, t_prime)
    print(f"t data generated for RG step {rg_step} and saved to {t_filename}")
    # if existing_t_file is not None and os.path.exists(existing_t_file):
    #     # Delete old files once done to prevent buildup
    #     os.remove(existing_t_file)
    print("-" * 100)
