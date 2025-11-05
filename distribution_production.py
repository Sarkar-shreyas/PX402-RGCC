"""Tools for generating and manipulating probability distributions in RG analysis.

This module provides functionality for creating initial distributions, sampling from
them, and performing various distribution manipulations required by the RG flow
analysis. It includes tools for phase generation, distribution centering, and
maintaining probability distribution invariants.
"""

import numpy as np
from config import Z_RANGE, Z_BINS


# ---------- Initial distribution helpers ---------- #
def generate_random_phases(N: int) -> np.ndarray:
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
        Array of shape (N, 4) containing random phases in [0, 2π].
    """
    phi_sample = np.random.uniform(0, 2 * np.pi, (N, 4))
    return phi_sample


def generate_initial_t_distribution(N: int) -> np.ndarray:
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
    g_sample = np.random.uniform(0, 1.0, N)
    t_dist = np.sqrt(g_sample)
    return t_dist


# ---------- Sampling helpers ---------- #
class Probability_Distribution:
    """A class for managing binned probability distributions in the RG analysis.

    This class provides tools for creating, analyzing, and sampling from
    probability distributions represented as normalized histograms. It supports
    operations needed for the RG flow analysis including distance calculations
    between distributions and statistical measurements.

    Parameters
    ----------
    values : numpy.ndarray
        Raw values to bin into a probability distribution.
    bins : int, optional
        Number of bins for the histogram (default from config.BINS).
    range : tuple, optional
        (min, max) range for binning (default from config.Z_RANGE).

    Attributes
    ----------
    bin_edges : numpy.ndarray
        Edges of the histogram bins, shape (bins+1,).
    histogram_values : numpy.ndarray
        Normalized histogram values, shape (bins,).
    cdf : numpy.ndarray
        Cumulative distribution function, shape (bins,).
    """

    def __init__(self, values, bins=Z_BINS, range=Z_RANGE):
        histogram_values, bin_edges = np.histogram(
            values, bins=bins, range=range, density=True
        )
        cdf = histogram_values.cumsum()
        cdf = cdf / cdf[-1]
        self.bin_edges = bin_edges
        self.domain_min = min(range)
        self.domain_max = max(range)
        self.domain_width = np.abs(self.domain_min) + np.abs(self.domain_max)
        self.bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        self.histogram_values = histogram_values
        self.cdf = cdf

    def histogram_distances(
        self, other_histogram_values, other_histogram_bin_edges
    ) -> float:
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
        integrand = (self.histogram_values - other_histogram_values) ** 2
        dz = np.diff(other_histogram_bin_edges)
        distance = float(np.sqrt(np.sum(integrand * dz)))
        return distance

    def update(self, new_hist_values, new_bin_edges):
        """Replace the stored histogram and recompute the CDF.

        Parameters
        ----------
        new_hist_values : numpy.ndarray
            Normalized histogram values for each bin (shape: (bins,)). The
            values should represent a probability density over the bin widths.
        new_bin_edges : numpy.ndarray
            Bin edges corresponding to the histogram (shape: (bins+1,)).

        Notes
        -----
        This method updates the distribution in-place and recomputes the
        cumulative distribution function (CDF) from the provided histogram
        values. If the provided histogram values are not normalized, the
        computed CDF will be scaled by their cumulative sum (i.e. cdf[-1]).
        The caller should ensure that the inputs are valid (non-negative and
        consistent shapes) to avoid runtime errors.
        """
        self.histogram_values = new_hist_values
        self.bin_edges = new_bin_edges
        cdf = new_hist_values.cumsum()
        cdf /= cdf[-1]
        self.cdf = cdf

    def mean_and_std(self) -> tuple:
        """Calculate mean and standard deviation of the distribution.

        Uses the normalized histogram values to compute first and second
        moments of the distribution.

        Returns
        -------
        tuple
            (mean, standard_deviation) of the distribution.
        """
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mean = (centers * self.histogram_values).sum() / self.histogram_values.sum()
        variance = (
            ((centers - mean) ** 2) * self.histogram_values
        ).sum() / self.histogram_values.sum()
        standard_deviation = np.sqrt(variance)
        return mean, standard_deviation

    def sample(self, N: int) -> np.ndarray:
        """Generate random samples from the distribution.

        Uses inverse transform sampling: generates uniform random numbers,
        maps them through the CDF, and interpolates within bins to get
        continuous values.

        Parameters
        ----------
        N : int
            Number of samples to generate.

        Returns
        -------
        numpy.ndarray
            Array of N samples drawn from the distribution.
        """
        # Inverse CDF method
        # u = np.random.uniform(0, 1.0, N)  # Uniform sample from 0 to 1
        # index = np.searchsorted(
        #     self.cdf, u
        # )  # Map it into our cdf histogram, will work fine because our cdf is normalised on initialisation.
        # index = np.clip(index, 0, len(self.cdf) - 1)  # Ensure we're within bounds
        # left_edge = self.bin_edges[
        #     index
        # ]  # Find the leftmost bin, could probably just set bin_edges[0]?
        # right_edge = self.bin_edges[
        #     index + 1
        # ]  # Find the rightmost bin, could probably just set bin_edges[-1]?

        # left_cdf = np.where(
        #     index == 0, 0.0, self.cdf[index - 1]
        # )  # Starting from the left, just set the first bin to 0 then move
        # right_cdf = self.cdf[index]  # Starting from the right, just follow indexing

        # # Add a guard in the denominator for extremely small values
        # denominator = np.maximum(right_cdf - left_cdf, 1e-15)
        # # Check how close to the right bin the value is
        # fraction = (u - left_cdf) / denominator
        # # Return the values mapped within their respective bins
        # return left_edge + fraction * (right_edge - left_edge)

        # Launder a.k.a rejection method
        # Get the bin widths, and total number of bins
        bin_width = np.diff(self.bin_edges)[0]
        num_bins = len(self.bin_centers)
        # Normalise the histogram manually
        normed = self.histogram_values / np.sum(self.histogram_values * bin_width)

        # Store the max height of the bins
        max_height = np.max(normed)
        # print(max_height)
        # Vectorise with numpy, run using reasonable batch sizes
        min_batch_size = 10000
        max_batch_size = 1000000
        # Track how many samples we've accepted and still need to be produced
        filled = 0
        remaining = N - filled
        # Placeholder array initialised early so we can just update values
        accepted = np.empty(N, dtype=float)
        num_iters = 0
        # Runs until we've got N samples
        while filled < N:
            num_iters += 1
            # Set the batch size to be between 10000 and 1000000, but use remaining if its in the bounds
            batch_size = max(min_batch_size, min(remaining, max_batch_size))
            # Random x and y draws within the domains of the existing dataset
            x = np.random.uniform(self.domain_min, self.domain_max, batch_size)
            y = np.random.uniform(0, max_height, batch_size)

            # Check which bin the x value falls into
            bin_number = np.ceil((x - self.domain_min) / bin_width).astype(int) - 1
            # Guard if we hit the boundaries
            bin_number = np.clip(bin_number, 0, num_bins - 1)

            # Store the heights at that bin
            heights = normed[bin_number]

            # Setup the y mask and slice the values to accept
            mask = y <= heights
            acceptable = x[mask]

            # Just try again if none are acceptable
            if len(acceptable) == 0:
                continue

            if num_iters % 1000 == 0:
                print(
                    f"Still laundering, Accepted: {len(acceptable)}, Remaining: {remaining}, batch size: {batch_size}"
                )
            # Only add how many we need, since we want exactly N samples
            to_accept = min(len(acceptable), remaining)
            # Fill the placeholder at those indices with the new accepted values
            # print(filled, to_accept)
            accepted[filled : filled + to_accept] = acceptable[:to_accept]
            filled += to_accept
            remaining -= to_accept

        return accepted


def center_z_distribution(Q_z: Probability_Distribution) -> Probability_Distribution:
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
    bin_edges = Q_z.bin_edges
    dz = np.diff(bin_edges)

    hist_values = Q_z.histogram_values
    symmetrised_Qz = 0.5 * (hist_values + hist_values[::-1])
    symmetrised_Qz /= np.sum(symmetrised_Qz * dz)
    Q_z.update(symmetrised_Qz, bin_edges)
    # centered_z = Q_z - np.median(Q_z)
    # new_z = np.concatenate([centered_z, -centered_z])
    return Q_z


def extract_t_samples(t: np.ndarray, N: int) -> np.ndarray:
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
    t_sample = t[np.random.randint(0, N, size=(N, 5))]

    return t_sample
