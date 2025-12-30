import numpy as np
from .rg_iterator import rg_iterator_for_nu
from .distribution_production import Probability_Distribution
from config import N, K, EXPRESSION
from scipy.stats import norm, linregress
from numpy.polynomial import polynomial
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Distribution manipulation helpers ---------- #
def estimate_z_peak(Q_z: Probability_Distribution) -> float:
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
    # start = time.time()
    # z_length = len(Q_z.histogram_values)
    # top_ten_percent = int(0.1 * z_length)
    # top_indices = np.argsort(Q_z.histogram_values)[-top_ten_percent:]
    # print(f"Sorted sample created in {time.time() - start:.3f} seconds.")

    # bin_values = Q_z.bin_centers[top_indices]
    # y_values = Q_z.histogram_values[top_indices]
    # if len(y_values) == 0:
    #     raise ValueError("The y values array is empty.")

    # length = np.random.permutation(len(y_values))
    # needed = np.array_split(length, 10)
    # subsets = [bin_values[needed[i]] for i in range(10)]

    # print("Fitting subsets")
    # params = [norm.fit(x) for x in subsets]
    # print(f"Fitting done in {time.time() - start:.3f} seconds")
    # if len(params) == 0:
    #     raise ValueError("No parameters were stored from the fit in estimate_z_peak.")

    # mus = [i for i, j in params]
    # return float(np.sum(mus) / 10)

    # Different approach, grows about center peak till 5% of probability mass is obtained. Used this in previous get_peak_from_subset code.
    bin_widths = np.diff(Q_z.bin_edges)  # Get widths
    bin_masses = Q_z.histogram_values * bin_widths  # Get masses

    # Check total mass, then calculate what 5% of that is.
    total_mass = np.sum(bin_masses)
    # print(f"Total bin mass of Q_z is {total_mass:.3f}")
    top_5_percent = 0.05 * total_mass

    # Store the bin indexes we care about - center and the 2 sides for later growth
    peak_bin = np.argmax(Q_z.histogram_values)
    left_bin = peak_bin
    right_bin = peak_bin
    final_bin = len(Q_z.histogram_values) - 1
    # Hold the mass of our current bin, will grow until 5%
    current_bin_mass = bin_masses[peak_bin]

    # We keep going until we hit 5%, or we hit the tails for whatever reason [thats a different problem then].
    while current_bin_mass < top_5_percent and (left_bin > 0 or right_bin < final_bin):
        # Now we decide whether to go left, or right. Check with some booleans
        move_left = left_bin > 0
        move_right = right_bin < final_bin

        # If both directions are safe, we'll decide by values.
        if move_left and move_right:
            left_val = Q_z.histogram_values[left_bin - 1]
            right_val = Q_z.histogram_values[right_bin + 1]
            # If left has higher or equivalent value, we'll move left. Slight bias choosing to go left if equivalent, but shouldn't matter too much.
            if left_val >= right_val:
                left_bin -= 1
                current_bin_mass += bin_masses[left_bin]
            else:
                right_bin += 1
                current_bin_mass += bin_masses[right_bin]
        elif move_left:
            # If only left is safe, of course we go left
            left_bin -= 1
            current_bin_mass += bin_masses[left_bin]
        else:
            # If only right is safe, of course we go right
            right_bin += 1
            current_bin_mass += bin_masses[right_bin]

    # print(f"{current_bin_mass:.3f} percent of bin mass accumulated.")

    # Now that we know which bins matter, we get their centers, and the z values at those edges.
    leftmost_bin = Q_z.bin_edges[left_bin]
    rightmost_bin = Q_z.bin_edges[right_bin + 1]
    # Use a sample from the distribution to prevent us from storing absurdly large amounts of raw data, maybe inaccurate but we'll see.
    samples = Q_z.sample(1 * (10**7))
    # And now we slice out the z values we need
    hist_mask = np.logical_and((samples >= leftmost_bin), (samples < rightmost_bin))
    top_5_percent_values = samples[hist_mask]
    # print(len(top_5_percent_values))
    # Shuffle the data so its randomly ordered
    np.random.shuffle(top_5_percent_values)
    # Now we split it into 10 equal sized subsets, shuffling before hand lets array_splits order slicing be fine.
    subsets = np.array_split(top_5_percent_values, 10)
    mu_guesses = [np.mean(subset) for subset in subsets]
    sigma_guesses = []
    fitted_mus = []
    for i in range(len(mu_guesses)):
        sigma_guesses.append(np.var(subsets[i]))

    for i in range(len(mu_guesses)):
        fitted_mus.append(norm.fit(subsets[i]))

    # print(fitted_mus)
    return float(np.mean(fitted_mus))


# ---------- Fitting helper ---------- #
def fit_z_peaks(x: np.ndarray, y: np.ndarray, method: str = "ls") -> tuple:
    """Fit a linear relationship between x and y data using different methods.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable data.
    y : numpy.ndarray
        Dependent variable data.
    method : str, optional
        Fitting method to use, by default "ls". Options are:
        - "ls": Custom least squares implementation
        - "linear": scipy.stats.linregress
        - "poly": numpy.polynomial.polynomial.Polynomial.fit

    Returns
    -------
    tuple
        A tuple containing (slope, r_squared):
        - slope: absolute value of the fitted slope
        - r_squared: coefficient of determination (R²)

    Raises
    ------
    KeyError
        If an invalid fitting method is specified.

    Notes
    -----
    All methods perform linear regression but use different implementations:
    - "ls" uses a manual least squares calculation
    - "linear" uses scipy's implementation
    - "poly" uses numpy's polynomial fitting
    """
    if method == "ls":
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x_mean - x) * (y_mean - y)) / np.sum(
            (x_mean - x) * (x_mean - x)
        )
        intercept = y_mean - slope * x_mean
        residual = y - slope * x - intercept
        ssr = float(np.dot(residual, residual))
        sst = float(np.dot(y - y_mean, y - y_mean))
        r2 = 1 - (ssr / sst)
        return float(np.abs(slope)), float(r2)

    elif method == "linear":
        result = linregress(x, y)
        slope = result.slope  # type: ignore
        r2 = result.rvalue**2  # type: ignore
        return float(np.abs(slope)), float(r2)
    elif method == "poly":
        passns, p = polynomial.Polynomial.fit(x, y, deg=1, full=True)
        resid = p[0]
        sst = float(np.dot(y, y))
        r2 = 1 - (resid / sst)  # type:ignore
        coef = np.polyfit(x, y, 1)
        return float(np.abs(coef[0])), float(r2)
    else:
        raise KeyError("An invalid fitting method was requested.")


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int = K) -> float:
    """Calculate critical exponent nu with the formula nu = ln(2^k)/ln(|slope|), where slope is calculated from fit_z_peaks, and k is the RG step number."""
    nu = np.log(2**rg_steps) / np.log(np.abs(slope))

    return float(nu)


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
    perturbation_list = np.array([0.003, 0.005, 0.007, 0.009, 0.011])
    num_perturbations = len(perturbation_list)

    # Set up an empty array to track z peaks
    z_peaks = np.zeros((K + 1, num_perturbations)).astype(float)
    unperturbed_z_peak = estimate_z_peak(fixed_point_Qz)

    print("-" * 100)
    print("Beginning z peak calculations")
    start_time = time.time()
    # Set up a perturbed sample of Z from the initial fixed point distribution
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_ylim([0, 0.3])
    # ax0.set_xlim([-5, 5])
    ax0.set_xlabel("z")
    ax0.set_ylabel("Q(z)")
    ax0.set_title("Q(z) vs z with z_0 = 0.007")

    # Plot unperturbed distribution
    ax0.plot(
        fixed_point_Qz.bin_centers, fixed_point_Qz.histogram_values, label="Unperturbed"
    )

    ax1.set_xlim([0, max(perturbation_list)])
    # ax1.set_ylim([0, 0.1])
    ax1.set_xlabel("z_0")
    ax1.set_ylabel("z_peak")
    ax1.set_title("z_peak vs z_0")

    z_sample = fixed_point_Qz.sample(N)

    for i, perturbation in enumerate(perturbation_list):
        perturbed_z = z_sample + perturbation
        sampled_Qz = fixed_point_Qz
        perturbed_Qz = Probability_Distribution(perturbed_z)

        # Store the first peak prior to any RG steps for each perturbation
        z_peaks[0, i] = np.abs(estimate_z_peak(perturbed_Qz) - unperturbed_z_peak)

        print(f"Performing RG step on perturbation {i}, z_0 = {perturbation:.5f}")
        # Perform RG iterations for the specific perturbation
        for n in range(1, K + 1):
            next_sampled_Qz = rg_iterator_for_nu(sampled_Qz)
            next_sampled_peak = estimate_z_peak(next_sampled_Qz)
            next_Qz = rg_iterator_for_nu(perturbed_Qz)
            peak = estimate_z_peak(next_Qz)
            peak_diff = peak - next_sampled_peak
            abs_peak_diff = np.abs(peak_diff)
            # next_z_sample = next_Qz.sample(N)
            print(
                f"Sampled peak: {next_sampled_peak:.5f}, Perturbed peak: {peak:.5f}, Absolute diff with perturbed peak: {abs_peak_diff:.5f}, Diff with sign: {peak_diff:.5f}"
            )
            z_peaks[n, i] = abs_peak_diff
            print(f"RG Step #{n} done for perturbation {i}")
            print(
                f"Time elapsed since analysis began: {time.time() - start_time:.3f} seconds"
            )
            if perturbation == 0.007 and n % 2 == 1:
                centers = 0.5 * (next_Qz.bin_edges[:-1] + next_Qz.bin_edges[1:])
                ax0.plot(
                    centers[::100],
                    next_Qz.histogram_values[::100],
                    label=f"RG step {n}",
                )
            perturbed_Qz = next_Qz
            sampled_Qz = next_sampled_Qz
            # perturbed_Qz = center_z_distribution(next_Qz)
            # sampled_Qz = center_z_distribution(next_sampled_Qz)

        print(
            f"All RG steps done for perturbation {i}. Time elapsed: {time.time() - start_time:.3f} seconds since beginning z peak calculations"
        )
        print("-" * 100)

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
        if n in set([1, 2, 5, 8]):
            ax1.scatter(perturbation_list, y, label=f"RG{n}")
        slope, r2 = fit_z_peaks(x, y, method="poly")
        # result = linregress(x, y)
        # slope = result.slope  # type: ignore
        # r2 = result.rvalue**2  # type: ignore
        nu = calculate_nu(slope, n)
        nu_estimates.append(nu)
        params.append({"Slope": slope, "R2": r2, "Nu": nu})
    ax0.legend()
    ax1.legend()
    plt.savefig(
        f"plots/{EXPRESSION}_Q(z)_perturbed_by_0.007_with_{N}_iters.png", dpi=150
    )
    plt.close()
    system_size = [2**i for i in range(len(nu_estimates))]
    plt.xlabel("2^n")
    plt.ylabel("Nu")
    plt.scatter(system_size, nu_estimates)
    plt.title("Nu against system size")
    plt.savefig(f"plots/{EXPRESSION}_Nu_{N}_iters.png", dpi=150)
    print("=" * 100)
    print(
        f"Analysis completed after {time.time() - current_time:.3f} seconds, returning results"
    )
    return {
        "Nu_values": nu_estimates,
        "parameters": params,
        "z_peaks": z_peaks.tolist(),
        "perturbations": perturbation_list.tolist(),
    }
