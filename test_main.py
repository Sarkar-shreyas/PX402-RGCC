"""Main execution script for the RG flow analysis.

This script orchestrates the full RG analysis workflow:
1. Find the fixed point distribution Q*(z) through iterative RG transformations
2. Estimate the critical exponent nu using perturbations around Q*(z)
3. Save results and convergence parameters to JSON files

The analysis uses parameters defined in config.py and saves results to the
params directory, separated into fixed point and critical exponent data.
"""

import numpy as np
from src.rg_iterator import rg_iterations_for_fp
from src.exponent_analysis import critical_exponent_estimation
from config import T_BINS
import time
from datetime import datetime, timezone
import json
import sys
import os

if __name__ == "__main__":
    start_time = time.time()
    N = int(sys.argv[1].strip())
    K = int(sys.argv[2].strip())
    output_dir = sys.argv[3].strip()
    print(
        f"Running program at {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    )
    fixed_point_Qz, fixed_point_Pt, params = rg_iterations_for_fp(N, T_BINS, K)
    final_params = list(params[-1])
    data = "".join(
        [
            f"Iteration #: {step}, Distance: {dist}, Std: {std}\n"
            for step, dist, std in params
        ]
    )
    fn = os.path.join(output_dir, f"Q_z_{N}_samples_{K}_steps.npz")
    np.savez_compressed(
        fn, hist=fixed_point_Qz.histogram_values, bins=fixed_point_Qz.bin_edges
    )
    estimation_params = critical_exponent_estimation(fixed_point_Qz)
    nu_values = estimation_params["Nu_values"]
    z_perturbations = estimation_params["perturbations"]
    z_p = [round(z_p, 4) for z_p in z_perturbations]
    z_peaks = estimation_params["z_peaks"]
    # print(data)

    print("-" * 100)
    with open(f"{output_dir}/final_params_with_{N}_samples.json", "w") as file:
        json.dump(final_params, file, indent=4)

    with open(f"{output_dir}/estimation_params_with_{N}_samples.json", "w") as file:
        json.dump(estimation_params, file, indent=4)

    print(
        f"Final values for FP determination: Distance between histograms: Distance between peaks = {final_params[1]:.4f}, Standard Deviation = {final_params[2]:.3f}."
    )
    print("-" * 100)
    print(f"Perturbations used: {z_p}")
    # print(f"Peaks used: {z_peaks}")
    print(f"Nu values obtained \n{nu_values}")
    print("=" * 100)
    end_time = time.time()
    print(f"Program took {end_time - start_time:.3f} seconds")
    print(
        f"Finished running program at {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    )
