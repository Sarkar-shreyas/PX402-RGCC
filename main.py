from rg_iterator import rg_iterations
from exponent_analysis import critical_exponent_estimation
from config import N, K, Z_PERTURBATION, BINS
import time

if __name__ == "__main__":
    start_time = time.time()
    fixed_point_Qz, fixed_point_Pt, params = rg_iterations(N, BINS, K)
    final_params = list(params[-1])
    data = "".join(
        [
            f"Iteration #: {step}, Distance: {dist}, Std: {std}\n"
            for step, dist, std in params
        ]
    )

    nu_estimate = critical_exponent_estimation(fixed_point_Qz, Z_PERTURBATION, K)
    # print(data)
    end_time = time.time()
    print("-" * 100)

    print(f"Program took {end_time - start_time:.3f} seconds")
    print(f"Estimation of nu: {nu_estimate:.3f}")
    print(
        f"Final values: Distance between histograms: Distance between peaks = {final_params[1]}, Standard Deviation = {final_params[2]}."
    )
