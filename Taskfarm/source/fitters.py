"""Fitting helpers and peak estimation utilities for RG analysis.

This module contains small helpers used to fit curves and extract peak
locations from binned z-histograms. It is intentionally lightweight so these
helpers can be imported into batch scripts without heavy dependencies.
"""

import numpy as np
from numpy.polynomial import polynomial

# from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from .utilities import launder


def std_derivative(
    rgs: np.ndarray | list, stds: np.ndarray | list, smoothing_factor: float
) -> np.ndarray | list:
    """Estimate the derivative of a standard-deviation curve using a spline.

    This helper smooths the input (rgs, stds) curve using a UnivariateSpline
    with smoothing parameter `s=smoothing_factor` and returns the derivative
    evaluated at the provided `stds` points.

    Parameters
    ----------
    rgs : array-like
        Independent variable values (e.g. RG step numbers or system sizes).
    stds : array-like
        Standard-deviation measurements corresponding to `rgs`.
    smoothing_factor : float
        Smoothing parameter passed to `scipy.interpolate.UnivariateSpline`.

    Returns
    -------
    numpy.ndarray
        Derivative values of the fitted spline evaluated at `stds`.
    """
    spline = UnivariateSpline(rgs, stds, s=smoothing_factor)
    derivative_line = spline.derivative()
    std_primes = derivative_line(stds)
    return std_primes


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float):
    """Simple gaussian for curve fit"""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def estimate_z_peak(
    z_hist: np.ndarray,
    z_bins: np.ndarray,
    z_bin_centers: np.ndarray,
) -> tuple:
    """Estimate peak location from a binned z-histogram using bootstrapped fitting.

    The function identifies the top 5% of bins by histogram height, resamples
    (via laundering) from those bins to obtain synthetic z samples, splits
    the synthetic sample into 10 subsets, fits a Gaussian to each subset via
    `scipy.stats.norm.fit`, and returns a tuple describing the spread of the
    fitted means.

    Parameters
    ----------
    z_hist : numpy.ndarray
        Histogram counts (not density) for the z distribution.
    z_bins : numpy.ndarray
        Histogram bin edges (length = n_bins + 1) corresponding to `z_hist`.
    z_bin_centers : numpy.ndarray
        Centers of the histogram bins (length = n_bins).

    Returns
    -------
    tuple
        (min_mean, max_mean, avg_mean) where `avg_mean` is the arithmetic mean
        of the 10 fitted Gaussian means and `min_mean`/`max_mean` are the
        mean ± std-dev of those fitted means (one-sigma bounds).

    Raises
    ------
    ValueError
        If the selected y-values are empty or no fit parameters are produced.

    Notes
    -----
    This function relies on `launder` to generate a continuous sample from
    the binned histogram and uses `scipy.stats.norm.fit` to estimate Gaussian
    parameters for each bootstrap subset.
    """
    # Restrict calculations within [-25,25]
    # z_min = -25.0 + shift
    # z_max = 25.0 + shift
    # mask = np.logical_and((z_bin_centers >= z_min), (z_bin_centers <= z_max))
    # z_hist = z_hist[mask]
    z_length = len(z_hist)
    top_five_percent = int(0.05 * z_length)
    top_indices = np.argsort(z_hist)[-top_five_percent:]
    top_indices = np.sort(top_indices)
    bin_centers = z_bin_centers[top_indices]
    bin_edges = z_bins[top_indices]
    # print(f"Min bin = {bin_edges[0]}, Max bin = {bin_edges[-1]}")
    y_values = z_hist[top_indices]
    sample = launder(10000000, y_values, bin_edges, bin_centers)
    if len(y_values) == 0:
        raise ValueError("The y values array is empty.")

    length = np.random.permutation(len(sample))
    subsets = np.array_split(sample[length], 10)
    # subsets = [bin_values[needed[i]] for i in range(10)]
    # print("Fitting subsets")
    params = [norm.fit(x) for x in subsets]
    if len(params) == 0:
        raise ValueError("No parameters were stored from the fit in estimate_z_peak.")

    mus = [i for i, j in params]
    std = np.std(mus, ddof=1)
    # min_mean = float(min(mus))
    # max_mean = float(max(mus))

    avg_mean = float(np.sum(mus) / 10)
    min_mean = float(avg_mean - std)
    max_mean = float(avg_mean + std)
    # print(f"Min bin = {bin_centers[0]}, Max bin = {bin_centers[-1]}")
    # print(f"Min mean = {float(min(mus))}, Max mean = {float(max(mus))}, std = {std}")
    return (min_mean, max_mean, avg_mean)

    # Different approach, grows about center peak till 5% of probability mass is obtained. Used this in previous get_peak_from_subset code.

    # Get densities
    # z_sample = launder(int(1e7), z_hist, z_bins, z_bin_centers)
    # plt.figure("test_fit")
    # z_density = get_density(z_hist, z_bins)

    # # Restrict calculations within [-25,25]
    # z_min = -25.0
    # z_max = 25.0
    # mask = np.logical_and((z_bin_centers >= z_min), (z_bin_centers <= z_max))

    # # Slice out the values within the window for analysis
    # centers = z_bin_centers[mask]
    # z_data = z_density[mask]

    # # Get bin width, single since bins are uniform
    # bin_width = np.diff(z_bins)[0]
    # bin_masses = z_data * bin_width  # Get masses

    # # Check total mass, then calculate what 5% of that is.
    # total_mass = np.sum(bin_masses)
    # # print(f"Total bin mass of Q_z is {total_mass:.3f}")
    # top_5_percent = 0.05 * total_mass

    # # Store the bin indexes we care about - center and the 2 sides for later growth
    # peak_bin = np.argmax(z_data)
    # left_bin = peak_bin
    # right_bin = peak_bin
    # final_bin = len(z_data) - 1
    # # Hold the mass of our current bin, will grow until 5%
    # current_bin_mass = bin_masses[peak_bin]

    # # We keep going until we hit 5%, or we hit the tails for whatever reason [thats a different problem then].
    # while current_bin_mass < top_5_percent and (left_bin > 0 or right_bin < final_bin):
    #     # Now we decide whether to go left, or right. Check with some booleans
    #     move_left = left_bin > 0
    #     move_right = right_bin < final_bin

    #     # If both directions are safe, we'll decide by values.
    #     if move_left and move_right:
    #         left_val = z_data[left_bin - 1]
    #         right_val = z_data[right_bin + 1]
    #         # If left has higher or equivalent value, we'll move left. Slight bias choosing to go left if equivalent, but shouldn't matter too much.
    #         if left_val >= right_val:
    #             left_bin -= 1
    #             current_bin_mass += bin_masses[left_bin]
    #         else:
    #             right_bin += 1
    #             current_bin_mass += bin_masses[right_bin]
    #     elif move_left:
    #         # If only left is safe, of course we go left
    #         left_bin -= 1
    #         current_bin_mass += bin_masses[left_bin]
    #     else:
    #         # If only right is safe, of course we go right
    #         right_bin += 1
    #         current_bin_mass += bin_masses[right_bin]

    # # print(f"{current_bin_mass:.3f} percent of bin mass accumulated.")

    # # Slice out the values corresponding to the top 5%
    # z_tip_values = centers[left_bin : right_bin + 1]

    # # density_tip_values = z_data[left_bin : right_bin + 1]
    # # tip_weights = density_tip_values * bin_width
    # tip_weights = bin_masses[left_bin : right_bin + 1]
    # # Compute weighted mean
    # mu = np.sum(z_tip_values * tip_weights) / np.sum(tip_weights)
    # print(
    #     f"The top 5 % has mean = {mu} with min = {z_tip_values[0]} and max = {z_tip_values[-1]}"
    # )
    # return float(mu)
    # bin_widths = np.diff(z_bins)  # Get widths
    # bin_masses = z_hist * bin_widths  # Get masses
    # z_density = get_density(z_hist, z_bins)
    # # z_min = -25.0
    # # z_max = 25.0
    # # mask = np.logical_and((z_bin_centers >= z_min), (z_bin_centers <= z_max))
    # # Slice out the values within the window for analysis
    # centers = z_bin_centers
    # z_data = z_density
    # bins = z_bins
    # # Get bin width, single since bins are uniform
    # bin_width = np.diff(bins)[0]
    # bin_masses = z_data * bin_width  # Get masses
    # # Check total mass, then calculate what 5% of that is.
    # total_mass = np.sum(bin_masses)
    # # print(f"Total bin mass of Q_z is {total_mass:.3f}")
    # top_5_percent = 0.05 * total_mass

    # # Store the bin indexes we care about - center and the 2 sides for later growth
    # peak_bin = np.argmax(z_data)
    # left_bin = peak_bin
    # right_bin = peak_bin
    # final_bin = len(z_data) - 1
    # # Hold the mass of our current bin, will grow until 5%
    # current_bin_mass = bin_masses[peak_bin]

    # # We keep going until we hit 5%, or we hit the tails for whatever reason [thats a different problem then].
    # while current_bin_mass < top_5_percent and (left_bin > 0 or right_bin < final_bin):
    #     # Now we decide whether to go left, or right. Check with some booleans
    #     move_left = left_bin > 0
    #     move_right = right_bin < final_bin

    #     # If both directions are safe, we'll decide by values.
    #     if move_left and move_right:
    #         left_val = z_data[left_bin - 1]
    #         right_val = z_data[right_bin + 1]
    #         # If left has higher or equivalent value, we'll move left. Slight bias choosing to go left if equivalent, but shouldn't matter too much.
    #         if left_val >= right_val:
    #             left_bin -= 1
    #             current_bin_mass += bin_masses[left_bin]
    #         else:
    #             right_bin += 1
    #             current_bin_mass += bin_masses[right_bin]
    #     elif move_left:
    #         # If only left is safe, of course we go left
    #         left_bin -= 1
    #         current_bin_mass += bin_masses[left_bin]
    #     else:
    #         # If only right is safe, of course we go right
    #         right_bin += 1
    #         current_bin_mass += bin_masses[right_bin]

    # # print(f"{current_bin_mass:.3f} percent of bin mass accumulated.")

    # # Now that we know which bins matter, we get their centers, and the z values at those edges.
    # leftmost_bin = bins[left_bin]
    # rightmost_bin = bins[right_bin + 1]
    # # Use a sample from the distribution to prevent us from storing absurdly large amounts of raw data, maybe inaccurate but we'll see.
    # samples = launder(1 * 10**7, z_data, bins, centers)
    # # And now we slice out the z values we need
    # hist_mask = np.logical_and((samples >= leftmost_bin), (samples < rightmost_bin))
    # top_5_percent_values = samples[hist_mask]
    # # print(len(top_5_percent_values))
    # # Shuffle the data so its randomly ordered
    # np.random.shuffle(top_5_percent_values)
    # # Now we split it into 10 equal sized subsets, shuffling before hand lets array_splits order slicing be fine.
    # subsets = np.array_split(top_5_percent_values, 10)
    # fitted_mus = []
    # for i in range(len(subsets)):
    #     mu, sigma = norm.fit(subsets[i])
    #     fitted_mus.append(float(mu))

    # # print(fitted_mus)
    # return float(np.mean(fitted_mus))


# ---------- Fitting helper ---------- #
def fit_z_peaks(x: np.ndarray, y: np.ndarray) -> tuple:
    """Fit a straight line to x vs y using a degree-1 polynomial and return slope and R².

    This function fits a first-degree polynomial to the provided `x` and
    `y` data (via `numpy.polynomial.Polynomial.fit` / `numpy.polyfit`) and
    computes the coefficient of determination (R²) from the residual.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable data.
    y : numpy.ndarray
        Dependent variable data.

    Returns
    -------
    tuple
        (abs_slope, r_squared) where `abs_slope` is the absolute value of the
        fitted slope and `r_squared` is the coefficient of determination.

    Notes
    -----
    The implementation uses `numpy.polynomial.Polynomial.fit` to obtain the
    residual and `numpy.polyfit` to extract the linear coefficient. This
    function assumes finite numeric data and does not perform input
    validation beyond relying on NumPy routines.
    """
    passns, p = polynomial.Polynomial.fit(x, y, deg=1, full=True)
    resid = p[0]
    sst = float(np.dot(y, y))
    r2 = 1 - (resid / sst)  # type:ignore
    coef = np.polyfit(x, y, 1)
    return float(np.abs(coef[0])), float(r2)
