#!/usr/bin/env python
"""Batch data-generation script for RG steps.

This script is intended to be executed from the command-line (or a job
script) to produce the next-step amplitude samples `t'` for a single RG
step. It supports two modes:

- Start from an initial analytic distribution (P(t)=2t-like) and generate
    samples of size `ARRAY_SIZE`.
- Continue from an existing `.npy` file containing prior samples.

Generated arrays are saved as `.npy` files to the specified output
directory. The script intentionally performs minimal in-process
post-processing so it can be run in parallel across job array tasks.

Usage
-----
See the `if __name__ == "__main__"` block for CLI usage details:
`data_generation.py ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP METHOD EXPR [EXISTING_T_FILE]`.
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
