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
from source.utilities import launder


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
    std_primes = derivative_line(rgs)
    return std_primes


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float):
    """Simple gaussian for curve fit"""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def estimate_z_peak(
    z_hist: np.ndarray,
    z_bins: np.ndarray,
    z_bin_centers: np.ndarray,
    rng: np.random.Generator,
    sampler: str,
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
    top_bin_indices = np.argsort(z_hist)[-top_five_percent - 1 :]
    top_indices = np.sort(top_indices)
    top_bin_indices = np.sort(top_bin_indices)
    bin_centers = z_bin_centers[top_indices]
    bin_edges = z_bins[top_bin_indices]
    # print(f"Min bin = {bin_edges[0]}, Max bin = {bin_edges[-1]}")
    y_values = z_hist[top_indices]
    sample = launder(10000000, y_values, bin_edges, bin_centers, rng, sampler)
    if len(y_values) == 0:
        raise ValueError("The y values array is empty.")
    overall_peak, _ = norm.fit(sample)
    length = rng.permutation(len(sample))
    subsets = np.array_split(sample[length], 10)
    # subsets = [bin_values[needed[i]] for i in range(10)]
    # print("Fitting subsets")
    params = [norm.fit(x) for x in subsets]
    if len(params) == 0:
        raise ValueError("No parameters were stored from the fit in estimate_z_peak.")

    mus = [i for i, j in params]
    # std = np.std(mus, ddof=1)
    min_mean = float(min(mus))
    max_mean = float(max(mus))
    # print(f"Min = {min_mean}, Max= {max_mean}, std = {std}")
    # avg_mean = float(np.sum(mus) / 10)
    # min_mean = float(avg_mean - std)
    # max_mean = float(avg_mean + std)
    # print(f"Min bin = {bin_centers[0]}, Max bin = {bin_centers[-1]}")
    # print(f"Min mean = {float(min(mus))}, Max mean = {float(max(mus))}, std = {std}")
    # print(f"Avg peak = {avg_mean}, Overall peak = {overall_peak}")
    return (min_mean, max_mean, overall_peak)


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
