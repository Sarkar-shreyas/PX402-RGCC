import numpy as np


# ---------- Initial distribution helpers ---------- #
def generate_random_phases(N: int) -> np.ndarray:
    """Generates an array of random phase values within [0, 2pi], returning an array of shape (N, 4)"""
    phi_sample = np.random.uniform(0, 2 * np.pi, (N, 4))
    return phi_sample


def generate_initial_t_distribution(N: int) -> np.ndarray:
    """Generates the initial distribution P(t) symmetric about t^2 = 0.5 and g = |t|^2"""
    g_sample = np.random.uniform(0, 1.0 + 1e-15, N)
    t_dist = np.sqrt(g_sample)
    return t_dist


# ---------- Sampling helpers ---------- #
def center_z_distribution(Q_z: np.ndarray) -> np.ndarray:
    """Helper function to re-center Q_k(z) after each rg iteration"""
    centered_z = Q_z - np.median(Q_z)
    new_z = np.concatenate([centered_z, -centered_z])
    return new_z


class Probability_Distribution:
    """A class to simplify manipulation of probability distributions for later use"""

    def __init__(self, values, bins):
        histogram_values, bin_edges = np.histogram(
            values, bins=bins, range=(-10.0, 10.0), density=True
        )
        cdf = histogram_values.cumsum()
        cdf = cdf / cdf[-1]
        self.bin_edges = bin_edges
        self.histogram_values = histogram_values
        self.cdf = cdf

    def histogram_distances(
        self, other_histogram_values, other_histogram_bin_edges
    ) -> float:
        """Returns the distance between two histograms following delta = int(sqrt(Q_k+1(z)^2 - Q_k(z)^2) dz)"""
        integrand = (self.histogram_values - other_histogram_values) ** 2
        dz = np.diff(other_histogram_bin_edges)
        distance = float(np.sqrt(np.sum(integrand * dz)))
        return distance

    def mean_and_std(self) -> tuple:
        """Return the mean and standard deviation of the distribution"""
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mean = (centers * self.histogram_values).sum() / self.histogram_values.sum()
        variance = (
            ((centers - mean) ** 2) * self.histogram_values
        ).sum() / self.histogram_values.sum()
        standard_deviation = np.sqrt(variance)
        return mean, standard_deviation

    def sample(self, N: int) -> np.ndarray:
        """Return an inverse CDF style sample of the probability distribution"""
        u = generate_initial_t_distribution(N)
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
    """Helper function to generate the (N,5) array of t values from the existing probability distribution"""
    t1 = P_t.sample(N)
    t2 = P_t.sample(N)
    t3 = P_t.sample(N)
    t4 = P_t.sample(N)
    t5 = P_t.sample(N)
    t_sample = np.stack([t1, t2, t3, t4, t5], axis=1)
    return t_sample
