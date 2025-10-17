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
def center_z_distribution(Q_z: np.ndarray) -> np.ndarray:
    """Center and symmetrize a Q(z) distribution.

    Centers the distribution around its median and enforces the required
    symmetry Q(z) = Q(-z) by including both positive and negative values.

    Parameters
    ----------
    Q_z : numpy.ndarray
        Array of z values from the current distribution.

    Returns
    -------
    numpy.ndarray
        Centered and symmetrized array of z values, twice the length of input.
    """
    centered_z = Q_z - np.median(Q_z)
    new_z = np.concatenate([centered_z, -centered_z])
    return new_z


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
