import numpy as np
from rg_iterator import rg_iterator_for_nu
from distribution_production import Probability_Distribution
from config import N, K, Z_RANGE, BINS
from scipy.stats import norm, linregress
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Distribution manipulation helpers ---------- #


def get_peak_from_subset(z_subset: np.ndarray) -> float:
    """Estimate the peak location of a Q(z) distribution subset using Shaw's method.

    The function builds a histogram of the provided subset, identifies the histogram
    bin with the maximum density, and grows a symmetric interval around that bin
    until approximately 5% of the total probability mass (in that neighbourhood) is
    captured. It then selects the raw z values that fall inside that interval and
    fits a Gaussian to those "tip" values using ``scipy.stats.norm.fit``. The
    returned value is the absolute value of the fitted mean (mu). If the fit
    produces a non-finite mean, the function falls back to a weighted average of
    the bin centres inside the selected interval.

    Parameters
    ----------
    z_subset : numpy.ndarray
        One-dimensional array of sampled z values representing a subset of the
        full Q(z) distribution. The array may be empty or contain very few
        elements; the function attempts to handle these cases gracefully.

    Returns
    -------
    float
        Estimated peak location (absolute value) for the subset. If the fit
        fails or produces non-finite results, a weighted average of bin centres
        is returned instead.
    """
    # Set up the histogram from the input subset
    z_values, bin_edges = np.histogram(z_subset, bins=BINS, range=Z_RANGE, density=True)
    dz = np.diff(bin_edges)

    # Find bin weights
    z_mass = z_values * dz

    # Find the indexes that would sort the bins, without sorting in place
    max_index = int(np.argmax(z_values))
    left_index = max_index
    right_index = max_index

    # Store the cumulative sum to check whether we've hit 5%
    cumulative_z_sum = z_mass[max_index]

    # Grow around the maximum value until 5% of probability mass is stored
    while cumulative_z_sum < 0.05 and (
        left_index > 0 or right_index < len(z_values) - 1
    ):
        # Set conditional to move left
        go_left = (right_index >= len(z_values) - 1) or (
            left_index > 0 and z_values[left_index - 1] >= z_values[right_index + 1]
        )

        # Go along the higher direction
        if go_left:
            left_index -= 1
            cumulative_z_sum += z_mass[left_index]
        else:
            right_index += 1
            cumulative_z_sum += z_mass[right_index]

    # Setup the slicing bounds for the z array
    z_low = bin_edges[left_index]
    z_high = bin_edges[right_index + 1]

    # Use values from the raw subset, not histogram data
    z_tip_values = z_subset[(z_subset >= z_low) & (z_subset < z_high)]

    # Use scipy's norm fit to apply a gaussian fit
    mu, _ = norm.fit(z_tip_values)
    # return float(mu)
    # Prevent infinite values messing up the log
    if not np.isfinite(mu):
        bin_centers = bin_edges[left_index : right_index + 1]
        mu = float(
            np.abs(
                np.average(
                    bin_centers,
                    weights=np.maximum(z_mass[left_index : right_index + 1], 1e-100),
                )
            )
        )

    return float(np.abs(mu))


def estimate_z_peak(z_sample: np.ndarray) -> float:
    """Estimate the average peak location for a full sample by aggregating subset peaks.

    The input sample is split into 10 equal (or near-equal) subsets. For each
    subset the ``get_peak_from_subset`` function is used to estimate a local
    peak. The arithmetic mean of these per-subset peak estimates is returned.

    Parameters
    ----------
    z_sample : numpy.ndarray
        One-dimensional array of sampled z values from a Probability_Distribution
        object. The array should contain enough samples to be split into the
        default 10 subsets; if it contains fewer elements, some subsets will be
        empty and ``get_peak_from_subset`` will handle them.

    Returns
    -------
    float
        Arithmetic mean of the per-subset peak estimates.
    """
    z_subsets = np.array_split(z_sample, 10)
    mu_values = [get_peak_from_subset(z_subset) for z_subset in z_subsets]

    return float(np.mean(mu_values))


# ---------- Critical Exponent estimation factory ---------- #
def critical_exponent_estimation(
    fixed_point_Qz: Probability_Distribution,
) -> dict:
    """Estimate critical exponent nu using RG flow analysis of perturbed distributions.

    This function implements a multi-step analysis to estimate the critical exponent nu:
    1. Applies a series of small perturbations to a fixed-point distribution Q(z)
    2. For each perturbation:
       - Tracks the evolution of distribution peaks through K RG steps
       - Uses ``estimate_z_peak`` to locate peaks in perturbed distributions
    3. Performs linear regression between initial perturbations and evolved peaks
    4. Estimates nu using the scaling relation nu = ln(2^k)/ln(z_k/z_0)

    The analysis includes visualization of the RG flow for z_0 = 0.007 and tracks
    computation time for each major step. A figure showing the evolution of Q(z)
    is saved to the plots directory.

    Parameters
    ----------
    fixed_point_Qz : Probability_Distribution
        The fixed-point distribution Q*(z) around which to perform perturbative
        analysis. This distribution should be at or very near the RG fixed point.

    Returns
    -------
    dict
        A dictionary containing:
        - 'Nu_values': List of nu estimates for each RG step
        - 'Nu_data': Dict with mean/median nu values and analysis bounds
        - 'parameters': List of dicts with slope and R² for each RG step
        - 'z_peaks': 2D array of peak locations [RG_step, perturbation]
        - 'perturbations': List of perturbation magnitudes used

    Notes
    -----
    The function uses a predefined set of perturbations from 4e-4 to 11e-4
    and averages nu estimates between RG steps 5 and 12 for the final result.
    Progress and timing information is printed to stdout during execution.
    """
    # Set up list of perturbations to try
    perturbation_list = np.array([4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 11e-4])
    num_perturbations = len(perturbation_list)

    # Set up an empty array to track z peaks
    z_peaks = np.zeros((K + 1, num_perturbations)).astype(float)

    print("-" * 100)
    print("Beginning z peak calculations")
    start_time = time.time()
    # Set up a perturbed sample of Z from the initial fixed point distribution
    plt.figure(figsize=(7, 4))
    plt.xlabel("z")
    plt.ylabel("Q(z)")
    plt.title("Q(z) vs z with z_0 = 0.007")
    plt.xlim([-3, 10])
    plt.ylim([0, 0.3])
    for i, perturbation in enumerate(perturbation_list):
        z_sample = fixed_point_Qz.sample(N)
        perturbed_z = z_sample + perturbation
        perturbed_Qz = Probability_Distribution(perturbed_z, BINS)

        # Store the first peak prior to any RG steps for each perturbation
        z_peaks[0, i] = estimate_z_peak(perturbed_Qz.sample(N))

        print(f"Performing RG step on perturbation {i}, z_0 = {perturbation:.5f}")
        # Perform RG iterations for the specific perturbation
        for n in range(1, K + 1):
            next_Qz = rg_iterator_for_nu(perturbed_Qz)
            next_z_sample = next_Qz.sample(N)
            z_peaks[n, i] = estimate_z_peak(next_z_sample)
            print(f"RG Step #{n} done for perturbation {i}")
            print(
                f"Time elapsed since analysis began: {time.time() - start_time:.3f} seconds"
            )
            if perturbation == 7e-4 and n % 2 == 1:
                centers = 0.5 * (next_Qz.bin_edges[:-1] + next_Qz.bin_edges[1:])
                plt.plot(centers, next_Qz.histogram_values, label=f"RG step {n}")
            perturbed_Qz = next_Qz

        print(
            f"All RG steps done for perturbation {i}. Time elapsed: {time.time() - start_time:.3f} seconds since beginning z peak calculations"
        )
        print("-" * 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/Q(z)_perturbed_by_0.007_with_{N}_iters.png", dpi=150)
    print("-" * 100)
    print(
        f"z peaks have been found. Time elapsed to complete calculations: {time.time() - start_time:.3f}"
    )
    print("Starting linear regression analysis")
    print("=" * 100)
    current_time = time.time()
    print(
        f"Analysis starting {current_time - start_time:.3f} seconds after beginning calculations"
    )
    # Find the estimation of nu for each perturbation and RG step taken
    nu_estimates = []
    params = []
    # plt.figure(figsize=(7, 4))
    # plt.xlabel("z_0")
    # plt.ylabel("z_n")
    # plt.xlim([0.0001, 0.0011])
    # plt.ylim([0, 0.15])
    # avg_z_peaks = [z_peaks[i, :].mean() for i in range(4, len(z_peaks))]
    # plt.plot(perturbation_list, avg_z_peaks)
    # plt.legend(labels=perturbation_list)
    # plt.savefig("z_n_against_z_0.png", dpi=150)

    for n in range(1, K + 1):
        print(
            f"Performing Nu estimation for RG step #{n} {time.time() - current_time:.3f} seconds after beginning analysis."
        )
        y = z_peaks[n, :].astype(float)
        x = np.array(perturbation_list).astype(float)

        # Slice x and y values to avoid infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        x_y = np.dot(x, y)
        x_x = np.dot(x, x)
        # result = linregress(x, y)
        # slope = result.slope  # type: ignore
        # r2 = result.rvalue**2  # type: ignore
        slope = x_y / x_x
        residual = y - slope * x
        sse = float(np.dot(residual, residual))
        sst = float(np.dot(y, y))
        r2 = 1 - (sse / sst)

        # Handle negative or infinite slopes
        if not np.isfinite(slope) or slope <= 0:
            nu_estimate = float("nan")
            nu_estimates.append(float(nu_estimate))
            # params.append(
            #     {"RG": n, "Slope": float(slope), "R2": float(result.rvalue**2)}  # type: ignore
            # )  # type: ignore
            params.append(
                {"RG": n, "Slope": float(slope), "R2": float(r2)}  # type: ignore
            )  # type: ignore
        else:
            # Estimate nu = log(2^n)/log(z_n/z_0) with slope = z_n/z_0
            nu_estimate = float(n * np.log(2) / np.log(slope))
            nu_estimates.append(float(nu_estimate))
            # params.append({"RG": n, "Slope": slope, "R2": float(result.rvalue**2)})  # type: ignore
            params.append({"RG": n, "Slope": slope, "R2": float(r2)})  # type: ignore

    start = 5
    end = 12
    nu_mean = float(np.mean(nu_estimates[start:end]))
    nu_median = float(np.median(nu_estimates[start:end]))
    nu_data = {"mean": nu_mean, "median": nu_median, "start": start, "end": end}
    print("=" * 100)
    print(
        f"Analysis completed after {time.time() - current_time:.3f} seconds, returning results"
    )
    return {
        "Nu_values": nu_estimates,
        "Nu_data": nu_data,
        "parameters": params,
        "z_peaks": z_peaks.tolist(),
        "perturbations": perturbation_list.tolist(),
    }
