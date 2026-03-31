"""Fitting helpers and peak estimation utilities for RG analysis.

This module implements the three-stage fitting pipeline used to extract the
critical exponent ν from RG Monte Carlo runs:

1. **Peak find** — `estimate_z_peak` locates the peak of the z-distribution
   produced by an EXP (perturbed) run by selecting the top-5% of histogram
   bins, resampling synthetic data via `launder`, and fitting `scipy.stats.norm`
   to 10 bootstrap subsets.

2. **Linear fit** — `fit_z_peaks` fits a straight line (degree-1 polynomial)
   to the log(shift) vs log(peak displacement) data collected across multiple
   EXP runs, returning the slope and R².

3. **ν extraction** — the slope from step 2 feeds into
   `utilities.calculate_nu`, which converts it to the critical exponent ν
   that characterises the divergence of the correlation length at the quantum
   phase transition: ξ ~ |δ|^{−ν}, where δ is the deviation from the
   critical point.

Gaussian fitting throughout uses `scipy.stats.norm.fit`.  The module is
intentionally lightweight so it can be imported into batch scripts without
heavy dependencies.
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

    Smooths the input (rgs, stds) curve with a `UnivariateSpline` and returns
    the derivative evaluated at each point in `rgs`.

    Args:
        rgs: Independent variable values (e.g. RG step numbers or system sizes).
        stds: Standard-deviation measurements corresponding to each value in `rgs`.
        smoothing_factor: Smoothing parameter ``s`` passed to
            `scipy.interpolate.UnivariateSpline`.  Larger values produce a
            smoother spline; ``s=0`` interpolates exactly.

    Returns:
        Derivative values of the fitted spline evaluated at `rgs`.
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
    """Estimate peak location from a binned z-histogram using bootstrapped Gaussian fitting.

    Algorithm (high level):

    1. Selects the top 5% of bins by raw count to focus on the peak region.
    2. Draws 10 M synthetic z-values from those bins via `launder` (inverse-CDF
       or rejection sampling, according to `sampler`).
    3. Randomly splits the synthetic sample into 10 bootstrap subsets.
    4. Fits `scipy.stats.norm` to each subset to obtain 10 candidate means.
    5. Returns the min/max of those means as uncertainty bounds alongside a
       single overall mean fitted to the full synthetic sample.

    Args:
        z_hist: Histogram counts (not density) for the z-distribution.
            Shape ``(n_bins,)``.
        z_bins: Bin edges of the histogram.  Shape ``(n_bins + 1,)``.
        z_bin_centers: Centre of each bin.  Shape ``(n_bins,)``.
        rng: NumPy random Generator used for synthetic resampling and subset
            permutation (must be a `numpy.random.Generator` instance).
        sampler: Resampling method passed to `launder`; e.g. ``"i"`` for
            inverse-CDF or ``"r"`` for rejection sampling.

    Returns:
        A 3-tuple ``(min_mean, max_mean, overall_peak)`` where:

        - ``min_mean`` — minimum of the 10 bootstrap-subset Gaussian means.
        - ``max_mean`` — maximum of the 10 bootstrap-subset Gaussian means.
        - ``overall_peak`` — Gaussian mean fitted to the full synthetic sample.

    Raises:
        ValueError: If the top-5% bin selection yields an empty count array,
            or if no fit parameters are returned from the bootstrap loop.
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
    """Fit a straight line to x vs y and return the slope magnitude and R².

    Used to extract the scaling relationship between log(perturbation shift) and
    log(peak displacement) across multiple EXP runs.  The returned slope feeds
    directly into `utilities.calculate_nu` to compute the critical exponent ν.

    Args:
        x: Independent variable data; typically log(shift) values for each EXP
            run.  Shape ``(n_points,)``.
        y: Dependent variable data; typically log(peak displacement) values
            corresponding to each entry in `x`.  Shape ``(n_points,)``.

    Returns:
        A 2-tuple ``(abs_slope, r_squared)`` where:

        - ``abs_slope`` — absolute value of the fitted linear slope.  Passed to
          `utilities.calculate_nu`; expected physical range ≈ 0.3–0.5 for
          IQHE/QSHE universality classes (yielding ν ≈ 2–3 after conversion).
        - ``r_squared`` — coefficient of determination R² for the linear fit.

    Notes:
        ν is extracted from the slope of log(peak displacement) vs
        log(perturbation shift).  A slope closer to 1 indicates a single
        scaling regime; deviations flag finite-size effects or multi-step
        corrections.  The implementation uses
        `numpy.polynomial.Polynomial.fit` for the residual and `numpy.polyfit`
        for the linear coefficient; both assume finite numeric input.
    """
    passns, p = polynomial.Polynomial.fit(x, y, deg=1, full=True)
    resid = p[0]
    sst = float(np.dot(y, y))
    r2 = 1 - (resid / sst)  # type:ignore
    coef = np.polyfit(x, y, 1)
    return float(np.abs(coef[0])), float(r2)
