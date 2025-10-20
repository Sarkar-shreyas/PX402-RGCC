"""Tools for generating and manipulating probability distributions in RG analysis.

This module provides functionality for creating initial distributions, sampling from
them, and performing various distribution manipulations required by the RG flow
analysis. It includes tools for phase generation, distribution centering, and
maintaining probability distribution invariants.
"""

import numpy as np
from config import Z_RANGE, BINS


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
    g_sample = np.random.uniform(0, 1.0 + 1e-15, N)
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

    def __init__(self, values, bins=BINS, range=Z_RANGE):
        histogram_values, bin_edges = np.histogram(
            values, bins=bins, range=range, density=True
        )
        cdf = histogram_values.cumsum()
        cdf = cdf / cdf[-1]
        self.bin_edges = bin_edges
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
        u = np.random.uniform(0, 1 + 1e-15, N)
        index = np.searchsorted(self.cdf, u)
        index = np.clip(index, 0, len(self.cdf) - 1)
        left_edge = self.bin_edges[index]
        right_edge = self.bin_edges[index + 1]

        left_cdf = np.where(index == 0, 0.0, self.cdf[index - 1])
        right_cdf = self.cdf[index]

        denominator = np.maximum(right_cdf - left_cdf, 1e-15)
        fraction = (u - left_cdf) / denominator
        return left_edge + fraction * (right_edge - left_edge)


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


def extract_t_samples(P_t: Probability_Distribution, N: int) -> np.ndarray:
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
    t1 = P_t.sample(N)
    t2 = P_t.sample(N)
    t3 = P_t.sample(N)
    t4 = P_t.sample(N)
    t5 = P_t.sample(N)

    t_sample = np.stack([t1, t2, t3, t4, t5], axis=1)
    return t_sample
