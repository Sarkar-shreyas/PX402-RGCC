import numpy as np
from numpy.polynomial import polynomial
from scipy.stats import norm

# ---------- Constants ---------- #
# N: int = 1 * (10**6)
K: int = 9
T_BINS: int = 1000
Z_BINS: int = 100000
Z_RANGE: tuple = (-50.0, 50.0)
Z_PERTURBATION: float = 0.007
DIST_TOLERANCE: float = 0.001
STD_TOLERANCE: float = 0.0005
T_RANGE: tuple = (1.39e-11, 1.0 - 1.39e-11)
EXPRESSION: str = "Shaw"
# EXPRESSION = "Shreyas"
# EXPRESSION = "Cain"
# EXPRESSION = "Jack"


# ---------- Saving utility ---------- #
def save_data(
    hist_vals: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray, filename: str
) -> None:
    np.savez_compressed(
        filename,
        histval=hist_vals,
        binedges=bin_edges,
        bincenters=bin_centers,
    )


# ---------- Data generators ---------- #


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


# ---------- t prime definition ---------- #
def generate_t_prime(t: np.ndarray, phi: np.ndarray) -> np.ndarray:
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
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)
    t3 = np.clip(t3, 0, 1)
    t4 = np.clip(t4, 0, 1)
    t5 = np.clip(t5, 0, 1)
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)

    if EXPRESSION == "Jack":
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
    elif EXPRESSION == "Shreyas":
        # My matrix (Blue)
        numerator = (r1 * t2 * (1 - np.exp(1j * phi4) * t3 * t4 * t5)) - (
            np.exp(1j * phi3)
            * r5
            * (r3 * t2 + np.exp(1j * (phi2 - phi1)) * r2 * r4 * t1 * t3)
        )

        denominator = (r3 - np.exp(1j * phi3) * r1 * r5) * (
            r3 - np.exp(1j * phi2) * r2 * r4
        ) + (t3 + np.exp(1j * phi4) * t4 * t5) * (t3 + np.exp(1j * phi1) * t1 * t2)
    elif EXPRESSION == "Cain":
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
    else:
        # Shaw's form (2023 thesis paper)
        numerator = (
            -(np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
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

    # t_prime = np.abs(
    #     numerator / np.where(np.abs(denominator) < 1e-12, np.nan + 0j, denominator)
    # )
    t_prime = np.abs(numerator) / np.abs(denominator)
    return t_prime
    # return np.clip(t_prime, 1.38e-11, 1 - 1.38e-11)


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
    tolerance = 1e-15
    g = np.clip(g, tolerance, 1 - tolerance)
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
    """Converts z data to t data directly, clipped within bounds corresponding to t in [1.38e-11, 1-1.38e-11], as mentioned by Shaw."""
    # z = np.clip(z, -25.0, 25.0)
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
    # t = np.clip(t, 1.39e-11, 1.0 - 1.39e-11)
    return np.log((1.0 / (t**2.0)) - 1.0)


# ---------- Histogram construction helper ---------- #
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

    def __init__(self, values, bins=Z_BINS, range=Z_RANGE, density: bool = True):
        histogram_values, bin_edges = np.histogram(
            values, bins=bins, range=range, density=density
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

    def update(self, new_hist_values):
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

            # if num_iters % 1000 == 0:
            #     print(
            #         f"Still laundering, Accepted: {len(acceptable)}, Remaining: {remaining}, batch size: {batch_size}"
            #     )
            # Only add how many we need, since we want exactly N samples
            to_accept = min(len(acceptable), remaining)
            # Fill the placeholder at those indices with the new accepted values
            # print(filled, to_accept)
            accepted[filled : filled + to_accept] = acceptable[:to_accept]
            filled += to_accept
            remaining -= to_accept

        return accepted


# ---------- Sampling helpers decoupled from P_D ---------- #
def launder(
    N: int, hist_vals: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray
) -> np.ndarray:
    """A copy of the laundering sample method decoupled from the Probability Distribution class"""
    # Launder a.k.a rejection method
    # Get the bin widths, and total number of bins
    bin_width = np.diff(bin_edges)[0]
    num_bins = len(bin_centers)
    # Normalise the histogram manually
    normed = hist_vals / np.sum(hist_vals * bin_width)

    # Store the max height of the bins
    max_height = np.max(normed)
    # Store the domain edges
    domain_min = bin_edges[0]
    domain_max = bin_edges[-1]
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
        x = np.random.uniform(domain_min, domain_max, batch_size)
        y = np.random.uniform(0, max_height, batch_size)

        # Check which bin the x value falls into
        bin_number = np.ceil((x - domain_min) / bin_width).astype(int) - 1
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
                f"Launder iteration {num_iters} - Accepted: {len(acceptable)}, Remaining: {remaining}, batch size: {batch_size}"
            )
        # Only add how many we need, since we want exactly N samples
        to_accept = min(len(acceptable), remaining)
        # Fill the placeholder at those indices with the new accepted values
        # print(filled, to_accept)
        accepted[filled : filled + to_accept] = acceptable[:to_accept]
        filled += to_accept
        remaining -= to_accept

    return accepted


def get_density(hist_vals: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    bin_counts = hist_vals.astype(float)
    bin_widths = np.diff(bin_edges)
    total = np.sum(bin_counts)
    probabilities = bin_counts / (total * bin_widths)
    return probabilities


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
    old_density = get_density(old_hist_val, old_bins)
    new_density = get_density(new_hist_val, new_bins)
    shaw_density_integrand = np.abs(new_density**2 - old_density**2)
    num_bins = len(old_hist_val)
    old_dz = np.diff(old_bins)
    density_shaw_distance = float(
        np.sum(np.sqrt(shaw_density_integrand) * old_dz) / num_bins
    )
    return density_shaw_distance


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


def center_z_distribution(z_hist: np.ndarray, z_bins: np.ndarray) -> np.ndarray:
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
    dz = np.diff(z_bins)
    symmetrised_z = 0.5 * (z_hist + z_hist[::-1])
    symmetrised_z /= np.sum(symmetrised_z * dz)
    return symmetrised_z


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
    # start = time.time()
    # z_length = len(Q_z.histogram_values)
    # top_ten_percent = int(0.1 * z_length)
    # top_indices = np.argsort(Q_z.histogram_values)[-top_ten_percent:]
    # print(f"Sorted sample created in {time.time() - start:.3f} seconds.")

    # bin_values = Q_z.bin_centers[top_indices]
    # y_values = Q_z.histogram_values[top_indices]
    # if len(y_values) == 0:
    #     raise ValueError("The y values array is empty.")

    # length = np.random.permutation(len(y_values))
    # needed = np.array_split(length, 10)
    # subsets = [bin_values[needed[i]] for i in range(10)]

    # print("Fitting subsets")
    # params = [norm.fit(x) for x in subsets]
    # print(f"Fitting done in {time.time() - start:.3f} seconds")
    # if len(params) == 0:
    #     raise ValueError("No parameters were stored from the fit in estimate_z_peak.")

    # mus = [i for i, j in params]
    # return float(np.sum(mus) / 10)

    # Different approach, grows about center peak till 5% of probability mass is obtained. Used this in previous get_peak_from_subset code.
    bin_widths = np.diff(z_bins)  # Get widths
    bin_masses = z_hist * bin_widths  # Get masses

    # Check total mass, then calculate what 5% of that is.
    total_mass = np.sum(bin_masses)
    # print(f"Total bin mass of Q_z is {total_mass:.3f}")
    top_5_percent = 0.05 * total_mass

    # Store the bin indexes we care about - center and the 2 sides for later growth
    peak_bin = np.argmax(z_hist)
    left_bin = peak_bin
    right_bin = peak_bin
    final_bin = len(z_hist) - 1
    # Hold the mass of our current bin, will grow until 5%
    current_bin_mass = bin_masses[peak_bin]

    # We keep going until we hit 5%, or we hit the tails for whatever reason [thats a different problem then].
    while current_bin_mass < top_5_percent and (left_bin > 0 or right_bin < final_bin):
        # Now we decide whether to go left, or right. Check with some booleans
        move_left = left_bin > 0
        move_right = right_bin < final_bin

        # If both directions are safe, we'll decide by values.
        if move_left and move_right:
            left_val = z_hist[left_bin - 1]
            right_val = z_hist[right_bin + 1]
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

    # Now that we know which bins matter, we get their centers, and the z values at those edges.
    leftmost_bin = z_bins[left_bin]
    rightmost_bin = z_bins[right_bin + 1]
    # Use a sample from the distribution to prevent us from storing absurdly large amounts of raw data, maybe inaccurate but we'll see.
    samples = launder(1 * 10**7, z_hist, z_bins, z_bin_centers)
    # And now we slice out the z values we need
    hist_mask = np.logical_and((samples >= leftmost_bin), (samples < rightmost_bin))
    top_5_percent_values = samples[hist_mask]
    # print(len(top_5_percent_values))
    # Shuffle the data so its randomly ordered
    np.random.shuffle(top_5_percent_values)
    # Now we split it into 10 equal sized subsets, shuffling before hand lets array_splits order slicing be fine.
    subsets = np.array_split(top_5_percent_values, 10)
    mu_guesses = [np.mean(subset) for subset in subsets]
    sigma_guesses = []
    fitted_mus = []
    for i in range(len(mu_guesses)):
        sigma_guesses.append(np.var(subsets[i]))

    for i in range(len(mu_guesses)):
        mu, sigma = norm.fit(subsets[i])
        fitted_mus.append(float(mu))

    # print(fitted_mus)
    return float(np.mean(fitted_mus))


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


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int = K) -> float:
    """Calculate critical exponent nu with the formula nu = ln(2^k)/ln(|slope|), where slope is calculated from fit_z_peaks, and k is the RG step number."""
    nu = np.log(2**rg_steps) / np.log(np.abs(slope))

    return float(nu)
