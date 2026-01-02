"""Utilities for RG analysis: sampling, conversions, I/O and statistics.

This module contains helpers for random-phase generation, initial
distributions, value transformations between t/g/z parametrizations,
histogram I/O and common statistical measures used by the RG pipeline.
"""
# Some mathematical expressions in this file are long by necessity; disable
# the line-length rule for this file to keep the formulas readable.
# flake8: noqa: E501

import numpy as np

# from time import time
from datetime import datetime, timezone


# ---------- Misc. utility ---------- #
def save_data(
    hist_vals: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray, filename: str
) -> None:
    """Save histogram arrays to a compressed .npz file.

    Parameters
    ----------
    hist_vals : numpy.ndarray
        Histogram counts (or densities) stored as `histval` in the archive.
    bin_edges : numpy.ndarray
        Bin edges array stored as `binedges` in the archive.
    bin_centers : numpy.ndarray
        Bin centers array stored as `bincenters` in the archive.
    filename : str
        Destination filename (will be written as a compressed `.npz`).
    """
    np.savez_compressed(
        filename,
        histval=hist_vals,
        binedges=bin_edges,
        bincenters=bin_centers,
    )


def get_current_date(format: str = "full") -> str:
    if format == "day":
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    elif format == "hour":
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H")
    elif format == "min":
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    else:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------- Data generators ---------- #
def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed=seed)


def generate_constant_array(N: int, value: float, M: int = 1) -> np.ndarray:
    """Generates a constant array of shape (N, M)"""
    if M == 1:
        return np.full(N, value, dtype=np.float64)
    else:
        return np.full(shape=(N, M), fill_value=value, dtype=np.float64)


def generate_random_phases(
    N: int,
    rng: np.random.Generator,
    i: int = 4,
) -> np.ndarray:
    """Generate random phase angles for RG transformation.

    Creates an array of uniformly distributed random phases in [0, 2π]
    used in the RG transformation step.

    Parameters
    ----------
    N : int
        Number of phase sets to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, i) containing random phases in [0, 2π].
    """
    phi_sample = rng.uniform(0, 2 * np.pi, (N, i))
    return phi_sample


def generate_initial_t_distribution(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate initial amplitude distribution P(t).

    Creates an initial distribution of amplitudes t with the property that
    the squared amplitudes g = |t|² are uniformly distributed in [0,1].
    This ensures P(t) is symmetric about t² = 0.5.

    Parameters
    ----------
    N : int
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        Array of N amplitude values t = √g where g ~ U[0,1].
    """
    g_sample = rng.uniform(0.0, 1.0, N)
    t_dist = np.sqrt(g_sample)
    return t_dist


def extract_t_samples(
    t: np.ndarray,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a matrix of amplitude samples for the RG transformation.

    Draws 5 independent sets of N samples from the given P(t) distribution
    and arranges them into a matrix suitable for the RG transformation step.

    Parameters
    ----------
    P_t : Probability_Distribution
        Distribution object representing the current P(t) distribution.
    N : int
        Number of sample sets to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 5) containing the sampled amplitude values.
    """
    t_sample = t[rng.integers(0, N, size=(N, 5))]
    return t_sample


# ---------- t prime computation ---------- #
def solve_matrix_eq(
    ts: np.ndarray, phis: np.ndarray, batch_size: int = 100000, output_index: int = 8
) -> np.ndarray:
    """
    Generates the A matrix from the given t, r and phi matrices
    """

    t1, t2, t3, t4, t5 = ts.T
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)
    phi12, phi15, phi23, phi31, phi34, phi42, phi45, phi53 = phis.T

    # Initialise a batch-size array of A and b to do the solve in batches
    A = np.zeros((batch_size, 10, 10), dtype=np.complex128)
    b = np.zeros((batch_size, 10, 1), dtype=np.float64)

    # Since it is initialised as a 3d array, we have to manually assign the values to indexes across every batch
    # Row 1
    A[:, 0, 0] = 1
    A[:, 0, 5] = -r1 * np.exp(1j * phi31)

    # Row 2
    A[:, 1, 1] = 1
    A[:, 1, 5] = t1 * np.exp(1j * phi31)

    # Row 3
    A[:, 2, 1] = -t2 * np.exp(1j * phi12)
    A[:, 2, 2] = 1
    A[:, 2, 7] = -r2 * np.exp(1j * phi42)

    # Row 4
    A[:, 3, 1] = -r2 * np.exp(1j * phi12)
    A[:, 3, 3] = 1
    A[:, 3, 7] = t2 * np.exp(1j * phi42)

    # Row 5
    A[:, 4, 2] = -r3 * np.exp(1j * phi23)
    A[:, 4, 4] = 1
    A[:, 4, 9] = -t3 * np.exp(1j * phi53)

    # Row 6
    A[:, 5, 2] = t3 * np.exp(1j * phi23)
    A[:, 5, 5] = 1
    A[:, 5, 9] = -r3 * np.exp(1j * phi53)

    # Row 7
    A[:, 6, 4] = t4 * np.exp(1j * phi34)
    A[:, 6, 6] = 1

    # Row 8
    A[:, 7, 4] = -r4 * np.exp(1j * phi34)
    A[:, 7, 7] = 1

    # Row 9
    A[:, 8, 0] = -t5 * np.exp(1j * phi15)
    A[:, 8, 6] = -r5 * np.exp(1j * phi45)
    A[:, 8, 8] = 1

    # Row 10
    A[:, 9, 0] = -r5 * np.exp(1j * phi15)
    A[:, 9, 6] = t5 * np.exp(1j * phi45)
    A[:, 9, 9] = 1

    # Assign b data
    b[:, 0, 0] = t1
    b[:, 1, 0] = r1

    x = np.linalg.solve(A, b)

    return x[:, output_index]


def generate_t_prime(
    t: np.ndarray, phi: np.ndarray, expression: str = "shaw"
) -> np.ndarray:
    """Generate next-step amplitudes using the RG transformation.

    Implements the core RG transformation that maps five input amplitudes and
    four phases to a new amplitude t'. The transformation preserves important
    symmetries while capturing the essential physics of the model.

    Parameters
    ----------
    t : numpy.ndarray
        Array of shape (N, 5) containing five amplitude samples per row.
        Values should be in range [0,1].
    phi : numpy.ndarray
        Array of shape (N, 4) containing four random phases per row.
        Values should be in range [0,2π].

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing the transformed amplitudes t'.
        Values are clipped to range [0,1-1e-15] for numerical stability.

    Notes
    -----
    The transformation includes safeguards against division by zero and
    produces values strictly less than 1 to prevent numerical issues in
    subsequent logarithmic transformations.
    """
    phi1, phi2, phi3, phi4 = phi.T
    t1, t2, t3, t4, t5 = t.T
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)

    if expression.strip().lower()[0] == "j":
        # Jack's form
        numerator = (
            -np.exp(1j * phi2) * r3 * t1 * t4
            - np.exp(1j * (phi3 + phi2)) * t2 * t4
            + np.exp(1j * (phi2 + phi3 - phi1)) * r1 * r5 * t2 * t3 * t4
            + t1 * t5
            + np.exp(1j * phi3) * r3 * t2 * t5
            + np.exp(1j * phi4) * r2 * r4 * t1 * t3 * t5
        )
        denominator = (
            1
            - np.exp(1j * (phi1 + phi4)) * r1 * r2 * r4 * r5
            + np.exp(1j * phi3) * r3 * t1 * t2
            + np.exp(1j * phi4) * r2 * r4 * t3
            - np.exp(1j * phi1) * r1 * r5 * t3
            - np.exp(1j * phi2) * r3 * t4 * t5
            - np.exp(1j * (phi2 + phi3)) * t1 * t2 * t4 * t5
        )
    elif expression.strip().lower()[0] == "c":
        # Cain's form (2005)
        numerator = (
            +t1 * t5 * (r2 * r3 * r4 * np.exp(1j * phi3) - 1)
            + t2
            * t4
            * (np.exp(1j * (phi1 + phi4)))
            * (r1 * r3 * r5 * np.exp(-1j * phi2) - 1)
            + t3 * (t2 * t5 * np.exp(1j * phi1) + t1 * t4 * np.exp(1j * phi4))
        )

        denominator = +(r3 - r2 * r4 * np.exp(1j * phi3)) * (
            r3 - r1 * r5 * np.exp(1j * phi2)
        ) + (t3 - t4 * t5 * np.exp(1j * phi4)) * (t3 - t1 * t2 * np.exp(1j * phi1))
    elif expression.strip().lower()[0] == "s":
        # Shaw's form (2023 thesis paper)
        numerator = (
            +(t1 * t5)
            - (np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
            + ((t2 * t4) * (np.exp(1j * (phi1 + phi4))))
            - (np.exp(1j * phi4) * t1 * t3 * t4)
            + (np.exp(1j * phi3) * r2 * r3 * r4 * t1 * t5)
            - (np.exp(1j * phi1) * t2 * t3 * t5)
        )
        denominator = (
            -1
            - (r2 * r3 * r4 * np.exp(1j * (phi3)))
            + (r1 * r3 * r5 * np.exp(1j * phi2))
            + (r1 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3)))
            + (t1 * t2 * t3 * np.exp(1j * phi1))
            - (t1 * t2 * t4 * t5 * np.exp(1j * (phi1 + phi4)))
            + (t3 * t4 * t5 * np.exp(1j * phi4))
        )
    elif expression.strip().lower()[0] == "t":
        # Shaw's second matrix (for Eq 2.13)
        numerator = (
            -t1 * t5
            + (np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
            - ((t2 * t4) * (np.exp(1j * (phi1 + phi4))))
            - (np.exp(1j * phi4) * t1 * t3 * t4)
            - (np.exp(1j * phi3) * r2 * r3 * r4 * t1 * t5)
            + (np.exp(1j * phi1) * t2 * t3 * t5)
        )
        denominator = (
            -1
            - (r2 * r3 * r4 * np.exp(1j * (phi3)))
            + (r1 * r3 * r5 * np.exp(1j * phi2))
            + (r1 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3)))
            + (t1 * t2 * t3 * np.exp(1j * phi1))
            - (t1 * t2 * t4 * t5 * np.exp(1j * (phi1 + phi4)))
            + (t3 * t4 * t5 * np.exp(1j * phi4))
        )
    else:
        raise ValueError("Invalid expression choice")

    t_prime = np.abs(numerator / denominator)
    return t_prime


def numerical_t_prime(ts: np.ndarray, phis: np.ndarray, N: int) -> np.ndarray:
    """
    A function to compute tprime numerically using np.linalg.solve. Defaults to using Shaw's matrix
    """
    num_batches = 20
    batch_size = N // num_batches
    tprime = np.empty(shape=(N, 1))
    for i in range(0, num_batches):
        index_slice = slice(i * batch_size, (i + 1) * batch_size)
        tprime[index_slice] = np.abs(
            solve_matrix_eq(ts[index_slice], phis[index_slice], batch_size)
        )

    return tprime


def rg_data_workflow(
    method: str, ts: np.ndarray, phis: np.ndarray, N: int, expr: str
) -> np.ndarray:
    """Perform the RG workflow based on method flag"""
    if method[0] == "a":  # Then we use the analytic form of tprime
        tprime = generate_t_prime(ts, phis, expr)
        return tprime
    elif method[0] == "n":
        tprime = numerical_t_prime(ts, phis, N)
        return tprime
    else:
        raise ValueError(f"Invalid method entered: {method}")


# ---------- Variable conversion helpers ---------- #
def convert_t_to_g(t: np.ndarray) -> np.ndarray:
    """Convert amplitude t to squared amplitude g.

    Computes g = |t|² for an array of complex or real amplitudes t.

    Parameters
    ----------
    t : numpy.ndarray
        Array of amplitude values to be squared.

    Returns
    -------
    numpy.ndarray
        Array of squared amplitudes g, same shape as input.
    """
    return np.abs(t) * np.abs(t)


def convert_g_to_z(g: np.ndarray) -> np.ndarray:
    """Convert squared amplitude g to RG flow parameter z.

    Computes z = ln((1-g)/g) with numerical stability enforced by clipping
    g values away from 0 and 1 to prevent divergences.

    Parameters
    ----------
    g : numpy.ndarray
        Array of squared amplitude values, should be in range [0,1].

    Returns
    -------
    numpy.ndarray
        Array of z values, same shape as input.

    Notes
    -----
    Input values are clipped to [1e-15, 1-1e-15] to ensure numerical stability
    of the logarithm.
    """
    return np.log((1.0 - g) / g)


def convert_z_to_g(z: np.ndarray) -> np.ndarray:
    """Convert RG flow parameter z back to squared amplitude g.

    Computes g = 1/(1 + exp(z)), the inverse transformation of convert_g_to_z.

    Parameters
    ----------
    z : numpy.ndarray
        Array of z values from the RG flow analysis.

    Returns
    -------
    numpy.ndarray
        Array of squared amplitudes g in range (0,1), same shape as input.
    """
    return 1.0 / (1.0 + np.exp(z))


def convert_z_to_t(z: np.ndarray) -> np.ndarray:
    """Convert z data to t data directly.

    The conversion is t = sqrt(1/(1 + exp(z))). Values are suitable for
    subsequent RG analysis; callers may clip `z` beforehand for numerical
    stability if required.
    """
    return np.sqrt(1.0 / (1.0 + np.exp(z)))


def convert_t_to_z(t: np.ndarray) -> np.ndarray:
    """Convert amplitude t directly to RG flow parameter z.

    Convenience function that combines convert_t_to_g and convert_g_to_z
    to perform the full t → g → z transformation.

    Parameters
    ----------
    t : numpy.ndarray
        Array of amplitude values to be converted.

    Returns
    -------
    numpy.ndarray
        Array of z values, same shape as input.
    """
    return np.log((1.0 / (t**2.0)) - 1.0)


# ---------- Sampling helpers decoupled from P_D ---------- #
def normalise_samplers(sampler: str) -> str:
    if sampler.strip().lower() in ("i", "inv", "cdf", "inverse"):
        return "i"
    elif sampler.strip().lower() in ("r", "rej", "reject", "rejection"):
        return "r"
    else:
        raise ValueError(f"Invalid sampling method entered: {sampler}")


def launder(
    N: int,
    hist_vals: np.ndarray,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    rng: np.random.Generator,
    sampler_input: str = "i",
) -> np.ndarray:
    """Perform laundering sampling decoupled from the ProbabilityDistribution class.

    The function performs inverse-CDF sampling from the provided binned
    histogram values to produce `N` continuous samples drawn from the
    histogram's implied distribution.
    """
    sampler = normalise_samplers(sampler_input)
    if sampler.strip().lower() == "i":
        return inverse_cdf_sampler(N, hist_vals, bin_edges, rng)
    elif sampler.strip().lower() == "r":
        return rejection_sampler(N, hist_vals, bin_edges, bin_centers, rng)
    else:
        raise KeyError("Invalid sampling method entered")


def inverse_cdf_sampler(
    N: int,
    hist_vals: np.ndarray,
    bin_edges: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # Inverse CDF method
    u = rng.random(size=N)
    densities = get_density(hist_vals, bin_edges)
    widths = np.diff(bin_edges)
    cdf = np.cumsum(densities * widths)
    # cdf = cdf / cdf[-1]

    # Map it into our cdf histogram
    index = np.searchsorted(cdf, u, side="right") - 1
    index = np.clip(index, 0, len(hist_vals) - 1)  # Ensure we're within bounds
    left_edge = bin_edges[index]
    right_edge = bin_edges[index + 1]

    # Check how close to the right bin the value is
    diff = right_edge - left_edge

    # Return values uniformly from their bins
    return left_edge + diff * rng.random(size=N)


def rejection_sampler(
    N: int,
    hist_vals: np.ndarray,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # Launder a.k.a rejection method
    bin_width = np.diff(bin_edges)[0]
    num_bins = len(bin_centers)
    normed = hist_vals / np.sum(hist_vals * bin_width)

    # Store the max height of the bins, and their edges
    max_height = np.max(normed)
    domain_min = bin_edges[0]
    domain_max = bin_edges[-1]

    # Vectorise with numpy, run using reasonable batch sizes. Use placeholders to track accepted/remaining quantity
    min_batch_size = 10000
    max_batch_size = 1000000
    filled = 0
    remaining = N - filled
    accepted = np.empty(N, dtype=float)
    num_iters = 0

    # Runs until we've got N samples
    while filled < N:
        num_iters += 1
        batch_size = max(min_batch_size, min(remaining, max_batch_size))
        # Random x and y draws within the domains of the existing dataset
        x = rng.uniform(domain_min, domain_max, batch_size)
        y = rng.uniform(0, max_height, batch_size)

        bin_number = np.ceil((x - domain_min) / bin_width).astype(int)
        bin_number = np.clip(bin_number, 0, num_bins - 1)

        # Store the heights at that bin
        heights = normed[bin_number]

        mask = y <= heights
        acceptable = x[mask]

        # Just try again if none are acceptable
        if len(acceptable) == 0:
            continue

        if num_iters % 1000 == 0:
            print(
                f"Launder iteration {num_iters} - Accepted: {len(acceptable)}, Remaining: {remaining}, batch size: {batch_size}"
            )

        # Only add how many we need, since we want exactly N samples
        to_accept = min(len(acceptable), remaining)
        accepted[filled : filled + to_accept] = acceptable[:to_accept]
        filled += to_accept
        remaining -= to_accept

    return accepted


def get_density(hist_vals: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Convert histogram counts into a probability density function.

    Parameters
    ----------
    hist_vals : numpy.ndarray
        Bin counts for the histogram.
    bin_edges : numpy.ndarray
        Bin edges array of length `len(hist_vals) + 1`.

    Returns
    -------
    numpy.ndarray
        Probability density (values such that integral over bins = 1).
    """
    bin_counts = hist_vals.astype(float)
    bin_widths = np.diff(bin_edges)
    total = np.sum(bin_counts)
    probabilities = bin_counts / (total * bin_widths)
    return probabilities


# ---------- Moments helpers ---------- #
def l2_distance(old_hist_val, new_hist_val, old_bins, new_bins) -> float:
    """Calculate L2 distance between this distribution and another.

    Computes the integrated squared difference between two normalized
    histograms: δ = √∫(Q_{k+1}(z)² - Q_k(z)²)dz.

    Parameters
    ----------
    other_histogram_values : numpy.ndarray
        Normalized values from other histogram, shape (bins,).
    other_histogram_bin_edges : numpy.ndarray
        Bin edges from other histogram, shape (bins+1,).

    Returns
    -------
    float
        L2 distance between the distributions.
    """
    # L2 distance between 2 histograms
    old_density = get_density(old_hist_val, old_bins)
    new_density = get_density(new_hist_val, new_bins)
    integrand = (new_density - old_density) ** 2
    dz = np.diff(old_bins)
    l2_distance = float(np.sqrt(np.sum(integrand * dz)))
    return l2_distance


def mean_squared_distance(old_hist_val, new_hist_val, old_bins, new_bins) -> float:
    """Compute Shaw's mean-squared distance between two histograms.

    The measure used in Shaw's workflow computes the mean over bins of the
    square-root of the positive part of (new_density^2 - old_density^2).

    Parameters
    ----------
    old_hist_val, new_hist_val : array-like
        Histogram counts for the old and new distributions respectively.
    old_bins, new_bins : array-like
        Corresponding bin-edge arrays.

    Returns
    -------
    float
        The computed mean squared (Shaw) distance.
    """
    # Shaw's MSD
    old_density = get_density(old_hist_val, old_bins)
    new_density = get_density(new_hist_val, new_bins)
    shaw_integrand = new_density**2 - old_density**2
    # Clip negative values that can occur when shifts are large
    shaw_integrand = np.clip(shaw_integrand, 0.0, None)
    return float(np.mean(np.sqrt(shaw_integrand)))


def hist_moments(hist_vals: np.ndarray, bins: np.ndarray) -> tuple:
    """Calculate mean and standard deviation of the distribution.

    Uses the normalized histogram values to compute first and second
    moments of the distribution.

    Returns
    -------
    tuple
        (mean, standard_deviation) of the distribution.
    """
    dz = np.diff(bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    probabilities = get_density(hist_vals, bins)
    mean = float(np.sum(probabilities * centers * dz))
    variance = float(np.sum((centers - mean) ** 2 * probabilities * dz))
    standard_deviation = np.sqrt(variance)
    return mean, standard_deviation


# ---------- Distribution manipulation helpers ---------- #
def center_z_distribution(
    z_hist: np.ndarray, z_bins: np.ndarray | None = None
) -> np.ndarray:
    """Symmetrize and renormalize a binned Q(z) distribution in-place.

    The function enforces the physical symmetry Q(z) = Q(-z) by averaging the
    histogram values with their reversed order and renormalising the result so
    the histogram integrates to unity over the bin widths. The underlying
    `Probability_Distribution` object is updated and returned.

    Parameters
    ----------
    Q_z : Probability_Distribution
        Distribution object whose histogram will be symmetrised and updated.

    Returns
    -------
    Probability_Distribution
        The same `Q_z` instance after its histogram and CDF have been
        symmetrised and renormalised.

    Notes
    -----
    The function assumes `Q_z.histogram_values` and `Q_z.bin_edges` are valid
    and will call `Q_z.update` to replace the stored histogram with the
    symmetrised version.
    """
    symmetrised_z = 0.5 * (z_hist + z_hist[::-1])
    return symmetrised_z


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int) -> float:
    """Calculate the critical exponent nu from a fitted slope.

    Parameters
    ----------
    slope : float
        Absolute slope obtained from a fit of z_peak vs perturbation.
    rg_steps : int, optional
        Number of RG steps (k) used in the scaling relation. Defaults to
        module-level `K`.

    Returns
    -------
    float
        Computed critical exponent nu using nu = ln(2^k)/ln(|slope|).
    """
    nu = np.log(2**rg_steps) / np.log(np.abs(slope))

    return float(nu)
