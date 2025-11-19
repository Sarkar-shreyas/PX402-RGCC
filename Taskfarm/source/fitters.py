import numpy as np
from numpy.polynomial import polynomial

# from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.stats import norm
from .utilities import get_density, launder


def std_derivative(
    rgs: np.ndarray | list, stds: np.ndarray | list, smoothing_factor: float
) -> np.ndarray | list:
    spline = UnivariateSpline(rgs, stds, s=smoothing_factor)
    derivative_line = spline.derivative()
    std_primes = derivative_line(stds)
    return std_primes


def _gauss(x: np.ndarray, a: float, mu: float, sigma: float):
    """Simple gaussian for curve fit"""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def estimate_z_peak(
    z_hist: np.ndarray, z_bins: np.ndarray, z_bin_centers: np.ndarray
) -> float:
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
    # z_length = len(z_hist)
    # top_ten_percent = int(0.1 * z_length)
    # top_indices = np.argsort(z_hist)[-top_ten_percent:]

    # bin_centers = z_bin_centers[top_indices]
    # bin_edges = z_bins[top_indices]
    # y_values = z_hist[top_indices]
    # sample = launder(10000000, y_values, bin_edges, bin_centers)
    # if len(y_values) == 0:
    #     raise ValueError("The y values array is empty.")

    # length = np.random.permutation(len(sample))
    # subsets = np.array_split(sample[length], 10)
    # # subsets = [bin_values[needed[i]] for i in range(10)]
    # # print("Fitting subsets")
    # params = [norm.fit(x) for x in subsets]
    # if len(params) == 0:
    #     raise ValueError("No parameters were stored from the fit in estimate_z_peak.")

    # mus = [i for i, j in params]
    # min_mean = min(mus)
    # max_mean = max(mus)
    # print(f"Min mu = {min_mean}, Max mu = {max_mean}")
    # return float(np.sum(mus) / 10)

    # Different approach, grows about center peak till 5% of probability mass is obtained. Used this in previous get_peak_from_subset code.

    # Get densities
    # z_sample = launder(int(1e7), z_hist, z_bins, z_bin_centers)
    z_density = get_density(z_hist, z_bins)

    # Restrict calculations within [-25,25]
    z_min = -25.0
    z_max = 25.0
    mask = np.logical_and((z_bin_centers >= z_min), (z_bin_centers <= z_max))

    # Slice out the values within the window for analysis
    centers = z_bin_centers[mask]
    z_data = z_density[mask]

    # Get bin width, single since bins are uniform
    bin_width = np.diff(z_bins)[0]
    bin_masses = z_data * bin_width  # Get masses

    # Check total mass, then calculate what 5% of that is.
    total_mass = np.sum(bin_masses)
    # print(f"Total bin mass of Q_z is {total_mass:.3f}")
    top_5_percent = 0.05 * total_mass

    # Store the bin indexes we care about - center and the 2 sides for later growth
    peak_bin = np.argmax(z_data)
    left_bin = peak_bin
    right_bin = peak_bin
    final_bin = len(z_data) - 1
    # Hold the mass of our current bin, will grow until 5%
    current_bin_mass = bin_masses[peak_bin]

    # We keep going until we hit 5%, or we hit the tails for whatever reason [thats a different problem then].
    while current_bin_mass < top_5_percent and (left_bin > 0 or right_bin < final_bin):
        # Now we decide whether to go left, or right. Check with some booleans
        move_left = left_bin > 0
        move_right = right_bin < final_bin

        # If both directions are safe, we'll decide by values.
        if move_left and move_right:
            left_val = z_data[left_bin - 1]
            right_val = z_data[right_bin + 1]
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

    # Slice out the values corresponding to the top 5%
    z_tip_values = centers[left_bin : right_bin + 1]
    density_tip_values = z_data[left_bin : right_bin + 1]
    tip_weights = density_tip_values * bin_width

    # Compute weighted mean
    mu = np.sum(z_tip_values * tip_weights) / np.sum(tip_weights)

    return float(mu)
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
    passns, p = polynomial.Polynomial.fit(x, y, deg=1, full=True)
    resid = p[0]
    sst = float(np.dot(y, y))
    r2 = 1 - (resid / sst)  # type:ignore
    coef = np.polyfit(x, y, 1)
    return float(np.abs(coef[0])), float(r2)
