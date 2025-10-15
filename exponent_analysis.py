import numpy as np
from rg_iterator import generate_t_prime, generate_random_phases
from distribution_production import Probability_Distribution, extract_t_samples
from utils import convert_z_to_g, convert_t_to_z
from config import N, K, Z_RANGE, Z_PERTURBATION, BINS
from scipy.stats import linregress


# ---------- Distribution manipulation helpers ---------- #
def get_peak_z_values(fixed_point_Qz_sample: np.ndarray) -> float:
    """
    Extracts the top 5% of values from the given Q_fp(z) distribution
    """
    # Set up the histogram from the input sample
    z_values, bin_edges = np.histogram(
        fixed_point_Qz_sample, bins=BINS, range=Z_RANGE, density=True
    )

    # Find bin weights
    bin_weights = z_values / z_values.sum()

    # Find the indexes that would sort the bins, without sorting in place
    ordered_weights_indexes = np.argsort(bin_weights)[::-1]

    # Store the cumulative sum to check whether we've hit 5%
    cumulative_z_sum = 0.0

    # Store our index choices
    top_5_percent = []

    # Go through our bin indexes and select+add them until we hit the top 5%
    for index in ordered_weights_indexes:
        top_5_percent.append(index)
        cumulative_z_sum += bin_weights[index]
        if cumulative_z_sum >= 0.05:
            break

    # Set up a mask to slice our z_values with
    z_mask = np.zeros_like(z_values).astype(bool)
    left_edges = bin_edges[top_5_percent]
    right_edges = bin_edges[np.array(top_5_percent) + 1]
    slice_condition = slice_condition = np.logical_and(
        z_values[:, None] >= left_edges, z_values[:, None] < right_edges
    )
    z_mask = np.any(slice_condition, axis=1)

    # Get the average of the top 5 percent, i.e the average peak value for later comparisons
    return z_values[z_mask].mean()


# ---------- Critical Exponent estimation factory ---------- #
def critical_exponent_estimation(
    fixed_point_Qz: Probability_Distribution,
    z_pertubration: float = Z_PERTURBATION,
    rg_steps: int = K,
) -> float:
    """
    Perturbs the fixed point distribution by a fixed z value,
    Calculates the average peak of the top 5% of the new distribution
    Performs linear regression to fit the peak against the perturbation
    Calculates v using v = ln(2^k)/ln(z_k/z_0)
    """

    # Set up a perturbed sample of Z from the initial fixed point distribution
    z_sample = fixed_point_Qz.sample(N)
    perturbed_z = z_sample + Z_PERTURBATION

    # Create the distribution from the perturbed sample
    perturbed_Qz = Probability_Distribution(perturbed_z, bins=BINS)

    # Store subsequent peaks for later analysis
    z_peaks = []

    for n in range(K + 1):
        # Use the average of 10 samples to get a robust peak value
        peaks_for_averaging = []
        for _ in range(10):
            new_z_sample = perturbed_Qz.sample(N // 10)
            avg_peak = get_peak_z_values(new_z_sample)
            peaks_for_averaging.append(avg_peak)

        # Find the average of the 10 samples for storage in our z_peaks list
        avg_10z_peaks = sum(peaks_for_averaging) / 10
        z_peaks.append(avg_10z_peaks)

        if n == K:
            break

        # Carry out an rg step
        perturbed_g = convert_z_to_g(perturbed_Qz.sample(N))
        perturbed_t = perturbed_g**0.5

        # Generate the distribution using perturbed t values
        perturbed_Pt = Probability_Distribution(perturbed_t, BINS)
        perturbed_t_sample = extract_t_samples(perturbed_Pt, N)
        phi_values = generate_random_phases(N)
        new_t_prime = generate_t_prime(perturbed_t_sample, phi_values)

        # Perform an rg step using the perturbed t' distribution, without re-centering z.
        perturbed_z = convert_t_to_z(new_t_prime)
        perturbed_Qz = Probability_Distribution(perturbed_z, BINS)

    # Perform a linear regression analysis on z_peak values against the z perturbation
    y_values = [np.log(np.abs(z_peak)) for z_peak in z_peaks if np.abs(z_peak) > 1e-14]
    x_values = list[range(len(y_values))]
    slope, intercept = linregress(x_values, y_values)
    nu_estimate: float = float(np.log(2.0) / slope)
    return nu_estimate
