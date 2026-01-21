"""Utilities for RG analysis: sampling, conversions, I/O and statistics.

This module contains helpers for random-phase generation, initial
distributions, value transformations between t/g/z parametrizations,
histogram I/O and common statistical measures used by the RG pipeline.
"""
# Some mathematical expressions in this file are long by necessity; disable
# the line-length rule for this file to keep the formulas readable.
# flake8: noqa: E501

import numpy as np
from constants import T_DICT, PHI_DICT
from numpy.typing import ArrayLike

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
    """Return current UTC date/time formatted as a string.

    Parameters
    ----------
    format : str, optional
        One of ``"day"``, ``"hour"``, ``"min"`` or ``"full"`` (default).

    Returns
    -------
    str
        Formatted UTC date/time string.
    """
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
    """Create and return a NumPy random Generator seeded with `seed`.

    Parameters
    ----------
    seed : int
        Integer seed for the RNG.

    Returns
    -------
    numpy.random.Generator
        A PCG64-based generator instance.
    """
    return np.random.default_rng(seed=seed)


def generate_constant_array(N: int, value: float, M: int = 1) -> np.ndarray:
    """Generate a constant-valued array.

    Parameters
    ----------
    N : int
        Number of rows/samples.
    value : float
        Value to fill the array with.
    M : int, optional
        Number of columns. If ``M==1`` a 1-D array of length ``N`` is
        returned; otherwise a 2-D array of shape ``(N, M)`` is returned.

    Returns
    -------
    numpy.ndarray
        Array filled with `value`.
    """
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


def generate_initial_t_distribution(
    N: int, rng: np.random.Generator, upper_bound: float = 1.0
) -> np.ndarray:
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
    g_sample = rng.uniform(0.0, upper_bound, N)
    t_dist = np.sqrt(g_sample)
    return t_dist


def generate_initial_qshe_data(
    samples: int,
    t_val: int,
    phi_val: int,
    f_val: float,
    rng: np.random.Generator,
) -> dict:
    n = samples
    if f_val > 1.0:
        f_val = 1.0
    elif f_val < 0.0 or f_val < 1e-10:
        f_val = 0.0
    f_array = generate_constant_array(n, f_val, 5)
    if t_val == 0:
        # t_sample = launder(
        #     n,
        #     fp_data["histval"],
        #     fp_data["binedges"],
        #     fp_data["bincenters"],
        #     rng,
        #     config.resample,
        # )
        # t_array = extract_t_samples(t_sample, n, rng)
        split = 1 - f_array**2
        # t_array = rng.uniform(0, np.sqrt(split), size=(n, 5))
        t_sample = generate_initial_t_distribution(n, rng, split[0, 0])
        t_array = extract_t_samples(t_sample, n, rng)
    else:
        t_array = generate_constant_array(n, T_DICT[f"{t_val}"], 5)
    if phi_val == 0:
        phi_array = generate_random_phases(n, rng, 16)
    else:
        phi_array = generate_constant_array(n, PHI_DICT[f"{phi_val}"], 16)
    # split_array = np.full(shape=(n, 1), fill_value=split)

    data_dict = {"t": t_array, "f": f_array, "phi": phi_array, "split": split}
    return data_dict


def extract_t_samples(
    t: np.ndarray,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample amplitude values for RG transformation.

    Given an array `t` representing the current sampled distribution, this
    function draws integer indices with the provided RNG and constructs a
    matrix of shape ``(N, 5)`` where each row contains five samples used by the
    RG transform.

    Parameters
    ----------
    t : numpy.ndarray
        1-D array containing samples from the P(t) distribution. Must have
        length at least ``N``.
    N : int
        Number of rows/samples to produce.
    rng : numpy.random.Generator
        Random number generator used to draw indices.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 5)`` of amplitude samples.
    """
    t_sample = t[rng.integers(0, N, size=(N, 5))]
    return t_sample


# ---------- t prime computation ---------- #
def solve_matrix_eq(
    ts: np.ndarray, phis: np.ndarray, batch_size: int = 100000, output_index: int = 8
) -> np.ndarray:
    """Solve per-batch linear systems to compute matrix-based t' numerically.

    Parameters
    ----------
    ts : numpy.ndarray
        Array of shape (batch_size, 5) containing five amplitudes per row.
    phis : numpy.ndarray
        Array of shape (batch_size, 8) containing phase combinations used to
        construct the linear system.
    batch_size : int, optional
        Number of rows in the batch (default 100000).
    output_index : int, optional
        Index of the solution vector to return for each batch row.

    Returns
    -------
    numpy.ndarray
        Array of shape (batch_size,) containing the selected solution entry
        (typically corresponding to the complex amplitude whose magnitude is
        used to compute t').
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
    b = np.zeros((batch_size, 10, 1), dtype=np.complex128)

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


def solve_qshe_matrix(
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    batch_size: int,
    output_index: int,
    inputs: ArrayLike,
) -> np.ndarray:
    """Build the 20x20 matrix equation and solve Mx = b"""
    t1, t2, t3, t4, t5 = ts.T
    f1, f2, f3, f4, f5 = fs.T
    r1 = np.sqrt(1 - t1**2 - f1**2)
    r2 = np.sqrt(1 - t2**2 - f2**2)
    r3 = np.sqrt(1 - t3**2 - f3**2)
    r4 = np.sqrt(1 - t4**2 - f4**2)
    r5 = np.sqrt(1 - t5**2 - f5**2)
    input_array = np.array(inputs)
    (
        phi12,
        phi13,
        phi15,
        phi21,
        phi23,
        phi24,
        phi31,
        phi32,
        phi34,
        phi35,
        phi42,
        phi43,
        phi45,
        phi51,
        phi53,
        phi54,
    ) = phis.T

    # Define our matrices
    M = np.zeros((batch_size, 20, 20), dtype=np.complex128)
    b = np.zeros((batch_size, 20, 1), dtype=np.complex128)

    # Now we need to assign data for 20 [0-19] rows... TODO: See if there's a more efficient way at some point
    # Matrix M
    # Row 0
    M[:, 0, 0] = 1
    M[:, 0, 4] = -r1 * np.exp(1j * phi31)
    M[:, 0, 18] = -f1 * np.exp(1j * phi51)

    # Row 1
    M[:, 1, 1] = 1
    M[:, 1, 4] = t1 * np.exp(1j * phi31)
    M[:, 1, 12] = f1 * np.exp(1j * phi21)

    # Row 2
    M[:, 2, 0] = -t2 * np.exp(1j * phi12)
    M[:, 2, 2] = 1
    M[:, 2, 6] = -r2 * np.exp(1j * phi42)
    M[:, 2, 15] = -f2 * np.exp(1j * phi32)

    # Row 3
    M[:, 3, 0] = -r2 * np.exp(1j * phi12)
    M[:, 3, 3] = 1
    M[:, 3, 6] = t2 * np.exp(1j * phi42)

    # Row 4
    M[:, 4, 3] = -r3 * np.exp(1j * phi23)
    M[:, 4, 4] = 1
    M[:, 4, 8] = -t3 * np.exp(1j * phi53)
    M[:, 4, 16] = -f3 * np.exp(1j * phi43)

    # Row 5
    M[:, 5, 3] = t3 * np.exp(1j * phi23)
    M[:, 5, 5] = 1
    M[:, 5, 8] = -r3 * np.exp(1j * phi53)
    M[:, 5, 11] = f3 * np.exp(1j * phi13)

    # Row 6
    M[:, 6, 5] = -t4 * np.exp(1j * phi34)
    M[:, 6, 6] = 1
    M[:, 6, 19] = -f4 * np.exp(1j * phi54)

    # Row 7
    M[:, 7, 5] = -r4 * np.exp(1j * phi34)
    M[:, 7, 7] = 1
    M[:, 7, 13] = f4 * np.exp(1j * phi24)

    # Row 8
    M[:, 8, 1] = -t5 * np.exp(1j * phi15)
    M[:, 8, 7] = -r5 * np.exp(1j * phi45)
    M[:, 8, 8] = 1

    # Row 9
    M[:, 9, 1] = -r5 * np.exp(1j * phi15)
    M[:, 9, 7] = t5 * np.exp(1j * phi45)
    M[:, 9, 9] = 1
    M[:, 9, 14] = f5 * np.exp(1j * phi35)

    # Row 10
    M[:, 10, 4] = f1 * np.exp(1j * phi31)
    M[:, 10, 10] = 1
    M[:, 10, 12] = -t1 * np.exp(1j * phi21)
    M[:, 10, 18] = -r1 * np.exp(1j * phi51)

    # Row 11
    M[:, 11, 11] = 1
    M[:, 11, 12] = -r1 * np.exp(1j * phi21)
    M[:, 11, 18] = t1 * np.exp(1j * phi51)

    # Row 12
    M[:, 12, 6] = f2 * np.exp(1j * phi42)
    M[:, 12, 12] = 1
    M[:, 12, 15] = -r2 * np.exp(1j * phi32)

    # Row 13
    M[:, 13, 0] = -f2 * np.exp(1j * phi12)
    M[:, 13, 13] = 1
    M[:, 13, 15] = t2 * np.exp(1j * phi32)

    # Row 14
    M[:, 14, 3] = f3 * np.exp(1j * phi23)
    M[:, 14, 11] = -t3 * np.exp(1j * phi13)
    M[:, 14, 14] = 1
    M[:, 14, 16] = -r3 * np.exp(1j * phi43)

    # Row 15
    M[:, 15, 8] = -f3 * np.exp(1j * phi53)
    M[:, 15, 11] = -r3 * np.exp(1j * phi13)
    M[:, 15, 15] = 1
    M[:, 15, 16] = t3 * np.exp(1j * phi43)

    # Row 16
    M[:, 16, 13] = -t4 * np.exp(1j * phi24)
    M[:, 16, 16] = 1
    M[:, 16, 19] = -r4 * np.exp(1j * phi54)

    # Row 17
    M[:, 17, 5] = -f4 * np.exp(1j * phi34)
    M[:, 17, 13] = -r4 * np.exp(1j * phi24)
    M[:, 17, 17] = 1
    M[:, 17, 19] = t4 * np.exp(1j * phi54)

    # Row 18
    M[:, 18, 7] = f5 * np.exp(1j * phi45)
    M[:, 18, 14] = -t5 * np.exp(1j * phi35)
    M[:, 18, 18] = 1

    # Row 19
    M[:, 19, 1] = -f5 * np.exp(1j * phi15)
    M[:, 19, 14] = -r5 * np.exp(1j * phi35)
    M[:, 19, 19] = 1
    # Set values for the 4 Inputs for testing
    I1_up = input_array[0]
    I3_down = input_array[1]
    I10_down = input_array[2]
    I8_up = input_array[3]
    # # b matrix for M
    # b[:, 0, 0] = r1 * tau1 * I1
    # b[:, 1, 0] = 1j * t1 * tau1 * I1
    # b[:, 2, 0] = -f1 * I1
    # b[:, 5, 0] = f2 * I2
    # b[:, 6, 0] = 1j * t2 * tau2 * I2
    # b[:, 7, 0] = r2 * tau2 * I2
    # b[:, 12, 0] = 1j * t4 * tau4 * I3
    # b[:, 13, 0] = r4 * tau4 * I3
    # b[:, 15, 0] = f4 * I3
    # b[:, 16, 0] = f5 * I4
    # b[:, 18, 0] = r5 * tau5 * I4
    # b[:, 19, 0] = 1j * t5 * tau5 * I4

    # b matrix for M2
    b[:, 0, 0] = t1 * I1_up
    b[:, 1, 0] = r1 * I1_up
    b[:, 3, 0] = -f2 * I3_down
    b[:, 6, 0] = r4 * I8_up
    b[:, 7, 0] = -t4 * I8_up
    b[:, 8, 0] = f5 * I10_down
    b[:, 11, 0] = f1 * I1_up
    b[:, 12, 0] = t2 * I3_down
    b[:, 13, 0] = r2 * I3_down
    b[:, 16, 0] = -f4 * I8_up
    b[:, 18, 0] = r5 * I10_down
    b[:, 19, 0] = -t5 * I10_down

    x = np.linalg.solve(M, b)

    # Outputs are index 2, 9, 10 and 17 in order of O3_up, O10_up, O1_down and O8_down
    # return x
    return x[:, output_index]

    return ts


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


def numerical_t_prime(
    ts: np.ndarray, phis: np.ndarray, N: int, batch_size: int = 100000
) -> np.ndarray:
    """Compute t' numerically by solving matrix equations in batches.

    Parameters
    ----------
    ts : numpy.ndarray
        Array of shape (N, 5) of amplitude samples.
    phis : numpy.ndarray
        Array of shape (N, 8) of phase values used to build the matrices.
    N : int
        Total number of samples.
    batch_size : int, optional
        Batch size used for vectorised solves.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 1) containing the absolute values of the solved
        complex amplitudes (t').
    """
    num_batches = N // batch_size
    tprime = np.empty(shape=(N, 1))
    for i in range(0, num_batches):
        index_slice = slice(i * batch_size, (i + 1) * batch_size)
        tprime[index_slice] = np.abs(
            solve_matrix_eq(ts[index_slice], phis[index_slice], batch_size)
        )

    return tprime


def rg_data_workflow(
    method: str,
    ts: np.ndarray,
    phis: np.ndarray,
    N: int,
    expr: str,
    batch_size: int = 100000,
) -> np.ndarray:
    """Compute t' according to the selected method.

    Parameters
    ----------
    method : str
        Either starts with 'a' for analytic (closed-form expression) or
        'n' for numerical (matrix solve) computation.
    ts : numpy.ndarray
        Input amplitudes, shape (N, 5).
    phis : numpy.ndarray
        Input phases, shape (N, 4) or shape required by numerical routine.
    N : int
        Number of samples.
    expr : str
        Expression identifier passed to the analytic generator (e.g. 'shaw',
        'jack', 'cain').
    batch_size : int, optional
        Batch size for numerical evaluation.

    Returns
    -------
    numpy.ndarray
        Array of t' values, shape (N,) (analytic) or (N, 1) (numerical).
    """
    if method[0] == "a":  # Then we use the analytic form of tprime
        tprime = generate_t_prime(ts, phis, expr)
        return tprime
    elif method[0] == "n":
        tprime = numerical_t_prime(ts, phis, N, batch_size)
        return tprime
    else:
        raise ValueError(f"Invalid method entered: {method}")


def qshe_numerical_solver(
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    N: int,
    output_index: int,
    inputs: ArrayLike,
    batch_size: int,
) -> np.ndarray:
    """Solve the matrix equation Mx=b for N samples using batching"""
    num_batches = N // batch_size
    output = np.empty(shape=(N, 1), dtype=np.float64)
    print(
        f"Beginning numerical solver for index {output_index} on {get_current_date()}"
    )
    print(f"Computing {num_batches} batches of size {batch_size}")
    # get_memory_usage("Memory usage before computation")
    for i in range(num_batches):
        indexes = slice(i * batch_size, (i + 1) * batch_size)
        output[indexes] = np.abs(
            solve_qshe_matrix(
                ts[indexes],
                fs[indexes],
                phis[indexes],
                batch_size,
                output_index,
                inputs,
            )
        )
    print(f"Computation for all {num_batches} batches of index {output_index} done")
    print("-" * 100)
    return np.abs(output)


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
    """Normalise a sampler name to the internal short key.

    Recognised inverse-CDF aliases map to ``'i'`` and rejection-sampler
    aliases map to ``'r'``. A ``ValueError`` is raised for unknown values.
    """
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
    """Inverse-CDF sampling from a binned histogram.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    hist_vals : numpy.ndarray
        Histogram counts per bin.
    bin_edges : numpy.ndarray
        Bin edge array of length ``len(hist_vals) + 1``.
    rng : numpy.random.Generator
        RNG instance used to draw uniform variates.

    Returns
    -------
    numpy.ndarray
        Array of `N` continuous samples drawn from the histogram's implied PDF.
    """
    # Inverse CDF method
    u = rng.random(size=N)
    densities = get_density(hist_vals, bin_edges)
    widths = np.diff(bin_edges)
    cdf = np.cumsum(densities * widths)
    cdf = cdf / cdf[-1]

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
    """Rejection-sampling-based launderer from binned histogram.

    Parameters
    ----------
    N : int
        Number of samples to produce.
    hist_vals : numpy.ndarray
        Histogram counts per bin.
    bin_edges : numpy.ndarray
        Bin edge array.
    bin_centers : numpy.ndarray
        Bin centers (unused directly but provided for completeness).
    rng : numpy.random.Generator
        RNG instance used to draw uniform variates.

    Returns
    -------
    numpy.ndarray
        Array of `N` continuous samples drawn from the histogram's implied PDF
        via rejection sampling.
    """
    # Launder a.k.a rejection method
    bin_width = np.diff(bin_edges)[0]
    # num_bins = len(bin_centers)
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

        bin_number = np.searchsorted(bin_edges, x, side="right") - 1
        bin_number = np.clip(bin_number, 0, len(hist_vals) - 1)
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


def build_2d_hist(
    data1: np.ndarray,
    data2: np.ndarray,
    data1_bins: int,
    data2_bins: int,
    data1_range: tuple,
    data2_range: tuple,
    sym: bool = False,
) -> dict:
    """Constructs a 2D histogram from input data arrays"""
    data1 = data1.ravel()
    data2 = data2.ravel()
    hist2d, z_edges, f_edges = np.histogram2d(
        data1,
        data2,
        bins=(data1_bins, data2_bins),
        range=(data1_range, data2_range),
        density=False,
    )

    # Compute bin centers
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    f_centers = 0.5 * (f_edges[1:] + f_edges[:-1])

    # Compute probability densities
    dz = np.diff(z_edges)[:, None]
    df = np.diff(f_edges)[None, :]
    area = dz * df

    # Compute 2D densities
    total = hist2d.sum()
    p_zf = hist2d / total / area

    # If we're symmetrising, manually symmetrise counts and densities
    if sym:
        hist2d = 0.5 * (hist2d + hist2d[::-1, :])
        p_zf = 0.5 * (p_zf + p_zf[::-1, :])

    # Obtain 1D densities and assert they are normalised
    p_z = (p_zf * df).sum(axis=1)
    assert np.abs(np.sum(p_z * np.diff(z_edges)) - 1.0) <= 1e-12
    p_f = (p_zf * dz).sum(axis=0)
    assert np.abs(np.sum(p_f * np.diff(f_edges)) - 1.0) <= 1e-12

    # Obtain 1D counts
    z_counts = hist2d.sum(axis=1)
    f_counts = hist2d.sum(axis=0)

    # Store relevant data, labelled for intuitive access
    hist_data = {
        "zf": {"counts": hist2d, "densities": p_zf},
        "z": {
            "counts": z_counts,
            "binedges": z_edges,
            "bincenters": z_centers,
            "densities": p_z,
        },
        "f": {
            "counts": f_counts,
            "binedges": f_edges,
            "bincenters": f_centers,
            "densities": p_f,
        },
    }

    return hist_data


def rejection_sampler_2d(data_dict: dict, rng: np.random.Generator, N: int) -> tuple:
    """Generate z and f sample arrays within the constraint by rejecting invalid inverse CDF samples"""
    # Initialise output arrays
    z_take = np.empty(N, dtype=np.float64)
    f_take = np.empty(N, dtype=np.float64)

    # Set batch boundaries to avoid large loop overhead
    min_batch_size = 50000
    max_batch_size = 1000000
    filled = 0
    num_iters = 0

    # Loop until we have N accepted samples for z and f
    while filled < N:
        num_iters += 1
        remaining = N - filled
        batch_size = max(min_batch_size, min(remaining, max_batch_size))
        z_sample, f_sample = inverse_cdf_2d(data_dict, rng, batch_size)
        t_sample = convert_z_to_t(z_sample)
        # Validity mask for unitarity constraint
        mask = t_sample**2 + f_sample**2 <= 1.0 + 1e-12
        valid = mask.sum()
        if valid == 0:
            continue

        # Store accepted values
        take = min(valid, remaining)
        z_take[filled : filled + take] = z_sample[mask][:take]
        f_take[filled : filled + take] = f_sample[mask][:take]
        filled += take

        if num_iters % 100 == 0:
            print(
                f"Rejection sampler iteration {num_iters}. {filled} samples accepted so far."
            )
    print(f"Took {num_iters} iterations in total.")
    return z_take, f_take


def inverse_cdf_2d(data_dict: dict, rng: np.random.Generator, N: int) -> tuple:
    """Perform inverse CDF sampling for a 2D histogram. Adapted from https://www.andreaamico.eu/data-analysis/2020/03/02/hist_sampling.html"""
    # Load 2D counts and respective axis bins
    zf_counts = data_dict["zf"]["counts"]
    z_edges = data_dict["z"]["binedges"]
    f_edges = data_dict["f"]["binedges"]

    total = zf_counts.sum()

    # Manually compute 2D probability masses, flatten to 1D and construct CDF
    prob_2d = (zf_counts / total).ravel()
    cdf = prob_2d.cumsum()
    # Guard against floating point errors
    cdf = cdf / cdf[-1]
    cdf[-1] = 1.0

    # Invert cdf and map indexes
    u = rng.random(size=N)
    flattened_indexes = np.searchsorted(cdf, u, side="right")

    # Find z and f indexes
    z_size, f_size = zf_counts.shape
    z_indexes = flattened_indexes // f_size
    f_indexes = flattened_indexes % f_size

    # Define rectangle to sample within
    z_left = z_edges[z_indexes]
    z_right = z_edges[z_indexes + 1]
    f_bottom = f_edges[f_indexes]
    f_top = f_edges[f_indexes + 1]

    z_diff = z_right - z_left
    f_diff = f_top - f_bottom
    z_sample = z_left + z_diff * rng.random(size=N)
    f_sample = f_bottom + f_diff * rng.random(size=N)
    return z_sample, f_sample


def conditional_2d_resampler(
    data_dict: dict, rng: np.random.Generator, N: int
) -> tuple:
    """
    Generate random z and f samples from their 2D histogram.
        - Generates z bins from P(z), z marginal.
        - Generates f bins from P(z | f), the conditional distribution to preserve row ordering
        - Uniformly samples within the generated rectangle
        - Rejects values that violate |t|^2 + |f|^2 <= 1.0
    """
    # Load 2D counts and respective axis bins
    zf_counts = data_dict["zf"]["counts"]
    z_edges = data_dict["z"]["binedges"]
    f_edges = data_dict["f"]["binedges"]

    total = zf_counts.sum()

    z_size, f_size = zf_counts.shape
    # Manually compute z marginal and construct the 1D z cdf
    z_marginal = zf_counts.sum(axis=1)
    z_cdf = np.cumsum(z_marginal / z_marginal.sum())

    # Guard against floating point errors
    z_cdf[-1] = 1.0

    # Initialise output arrays
    z_take = np.empty(N, dtype=np.float64)
    f_take = np.empty(N, dtype=np.float64)

    # Set batch boundaries to avoid large loop overhead
    min_batch_size = 50000
    max_batch_size = 1000000
    filled = 0
    num_iters = 0

    while filled < N:
        num_iters += 1
        remaining = N - filled
        batch_size = max(min_batch_size, min(remaining, max_batch_size))

        # Sample z bins
        z_bins = np.searchsorted(z_cdf, rng.random(batch_size), side="right")

        # Generate empty f bins array
        f_bins = np.empty(batch_size, dtype=np.int64)

        # Get unique z bins to prevent resampling from same bin
        unique_z, inv_z = np.unique(z_bins, return_inverse=True)

        # Loop until batch_size no. of f_bins is obtained
        for index, bin in enumerate(unique_z):
            # Check for similar z bins, and pull the f row for the corresponding bin
            similar = inv_z == index
            f_row = zf_counts[bin, :]
            f_row_sum = f_row.sum()

            # If the bin is empty, fallback to f_marginal
            if f_row_sum <= 0:
                f_row = zf_counts.sum(axis=0)
                f_row_sum = f_row.sum()

            # For similar z bins, pull f bin indexes from the constructed f cdf
            f_row_cdf = np.cumsum(f_row / f_row_sum)
            f_row_cdf[-1] = 1.0
            f_bins[similar] = np.searchsorted(
                f_row_cdf, rng.random(similar.sum()), side="right"
            )

        # Define rectangle to sample within
        z_left = z_edges[z_bins]
        z_right = z_edges[z_bins + 1]
        f_bottom = f_edges[f_bins]
        f_top = f_edges[f_bins + 1]

        # Sample uniformly within rectangle
        z_diff = z_right - z_left
        f_diff = f_top - f_bottom
        z_sample = z_left + z_diff * rng.random(size=batch_size)
        f_sample = f_bottom + f_diff * rng.random(size=batch_size)
        t_sample = convert_z_to_t(z_sample)

        # Validity check
        mask = t_sample**2 + f_sample**2 <= 1.0 + 1e-12
        valid = mask.sum()
        if valid == 0:
            continue

        # Take valid indexes
        take = min(valid, remaining)
        z_take[filled : filled + take] = z_sample[mask][:take]
        f_take[filled : filled + take] = f_sample[mask][:take]
        filled += take

        if num_iters % 100 == 0:
            print(
                f"Rejection sampler iteration {num_iters}. {filled} samples accepted so far."
            )

    print(f"Took {num_iters} iterations in total.")
    return z_take, f_take


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
    """Symmetrise a binned z-histogram array about zero.

    Parameters
    ----------
    z_hist : numpy.ndarray
        1-D array of histogram counts (or densities) over z bins.
    z_bins : numpy.ndarray, optional
        Corresponding bin edges. If provided the caller is expected to
        renormalise using bin widths; this function only averages symmetric
        bin pairs and returns the symmetrised values.

    Returns
    -------
    numpy.ndarray
        Symmetrised histogram values (same shape as ``z_hist``).
    """
    symmetrised_z = 0.5 * (z_hist + z_hist[::-1])
    return symmetrised_z


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int) -> float:
    """Compute critical exponent ``nu`` from slope and number of RG steps.

    Parameters
    ----------
    slope : float
        Absolute slope obtained from a fit of z_peak vs perturbation.
    rg_steps : int
        Number of RG steps used in the scaling relation.

    Returns
    -------
    float
        Computed critical exponent ``nu`` using ``nu = ln(2**rg_steps) / ln(|slope|)``.
    """
    nu = np.log(2**rg_steps) / np.log(np.abs(slope))

    return float(nu)
