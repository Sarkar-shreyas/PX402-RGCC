#!/usr/bin/env python
"""
This file will handle the data generation for every RG step. It will generate arrays in batches of size N, looped within the slurm script
"""

import os
import sys
import numpy as np
from .utilities import (
    generate_initial_t_distribution,
    generate_random_phases,
    generate_t_prime,
    extract_t_samples,
    convert_t_to_g,
    convert_t_to_z,
)

if __name__ == "__main__":
    # Load input params, checking if we're starting RG steps or continuing from an input sample
    if len(sys.argv) == 5:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        existing_t_file = "None"
    elif len(sys.argv) == 6:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        existing_t_file = sys.argv[5].strip()
    else:
        raise SystemExit(
            "Usage: histogram_manager.py ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP [EXISTING_T_FILE]"
        )

    print("-" * 100)
    print(f"Beginning data generation for RG step {rg_step}")

    if initial == 1:
        t = generate_initial_t_distribution(array_size)
        t_initial_mean = np.mean(t)
        print(f"Generated initial t distribution with mean: {t_initial_mean}")
    else:
        print(f"Using t data from {existing_t_file}")
        t = np.load(existing_t_file)
    phases = generate_random_phases(array_size)
    t_array = extract_t_samples(t, array_size)
    t_prime = generate_t_prime(t_array, phases)
    g = convert_t_to_g(t_prime)
    z = convert_t_to_z(t_prime)
    t_filename = os.path.join(
        output_dir, f"t_data_RG{rg_step}_{array_size}_samples.npy"
    )
    g_filename = os.path.join(
        output_dir, f"g_data_RG{rg_step}_{array_size}_samples.npy"
    )
    z_filename = os.path.join(
        output_dir, f"z_data_RG{rg_step}_{array_size}_samples.npy"
    )
    np.save(t_filename, t_prime)
    np.save(g_filename, g)
    np.save(z_filename, z)
    print(f"t data generated for RG step {rg_step} and saved to {t_filename}")
    print(f"g data generated for RG step {rg_step} and saved to {g_filename}")
    print(f"z data generated for RG step {rg_step} and saved to {z_filename}")

    # if existing_t_file is not None and os.path.exists(existing_t_file):
    #     # Delete old files once done to prevent buildup
    #     os.remove(existing_t_file)
    print("-" * 100)
