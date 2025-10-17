"""Main execution script for the RG flow analysis.

This script orchestrates the full RG analysis workflow:
1. Find the fixed point distribution Q*(z) through iterative RG transformations
2. Estimate the critical exponent nu using perturbations around Q*(z)
3. Save results and convergence parameters to JSON files

The analysis uses parameters defined in config.py and saves results to the
params directory, separated into fixed point and critical exponent data.
"""

from rg_iterator import rg_iterations_for_fp
from exponent_analysis import critical_exponent_estimation
from config import N, K, BINS
import time
import json

if __name__ == "__main__":
    start_time = time.time()
    fixed_point_Qz, fixed_point_Pt, params = rg_iterations_for_fp(N, BINS, K)
    final_params = list(params[-1])
    data = "".join(
        [
            f"Iteration #: {step}, Distance: {dist}, Std: {std}\n"
            for step, dist, std in params
        ]
    )

    estimation_params = critical_exponent_estimation(fixed_point_Qz)
    nu_data = estimation_params["Nu_data"]
    nu_values = estimation_params["Nu_values"]
    final_nu = nu_data["mean"]
    z_perturbations = estimation_params["perturbations"]
    z_p = [round(z_p, 4) for z_p in z_perturbations]
    z_peaks = estimation_params["z_peaks"]
    # print(data)

    print("-" * 100)
    with open(
        f"params/fixed_point/final_params_with_{N}_samples.json", "w", encoding="utf-8"
    ) as file:
        json.dump(final_params, file, indent=4)

    with open(
        f"params/exponent/estimation_params_with_{N}_samples.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(estimation_params, file, indent=4)

    print(
        f"Final values for FP determination: Distance between histograms: Distance between peaks = {final_params[1]:.4f}, Standard Deviation = {final_params[2]:.3f}."
    )
    print("-" * 100)
    print(f"Perturbations used: {z_p}")
    # print(f"Peaks used: {z_peaks}")
    print(f"Nu values obtained \n{nu_values}")
    print(f"Final nu estimation: {final_nu:.3f}")
    print("=" * 100)
    end_time = time.time()
    print(f"Program took {end_time - start_time:.3f} seconds")
