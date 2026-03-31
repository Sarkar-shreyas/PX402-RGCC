"""Authoritative RG engine: MC sampling, transformations, histograms, and statistics.

This module implements the core mathematics of the Renormalization Group (RG) pipeline.
It is the single source of truth for all sample generation, RG transformation logic,
variable conversions, and convergence statistics.  Do not modify physics logic or
algorithm implementations in this file.

RG transformation variants
---------------------------
Two computation paths are supported:

- **Analytic** (``method[0] == 'a'``): closed-form 4-phase expressions for t'.
  Four formula variants are available: ``"jack"``, ``"cain"``, ``"shaw"`` (default),
  and ``"t"`` (Shaw's second matrix, Eq 2.13).  Selected via ``generate_t_prime``.
- **Numerical** (``method[0] == 'n'``): solves a 10×10 (IQHE) or 20×20 (QSHE)
  complex linear system Ax = b per sample, processed in memory-bounded batches.
  Entry points: ``solve_matrix_eq`` / ``numerical_t_prime`` (IQHE) and
  ``solve_qshe_matrix`` / ``qshe_numerical_solver`` (QSHE).

Key physics variables
----------------------
- ``t`` — transmission amplitude, real-valued, domain [0, 1].
- ``r`` — complementary amplitude, ``r = √(1 − t²)``, domain [0, 1].
- ``g`` — squared amplitude, ``g = |t|²``, domain [0, 1].
- ``z`` — RG flow parameter (log-ratio), ``z = ln((1−g)/g)``, domain ℝ.
  At the critical fixed point the z-distribution is symmetric about z = 0.
- ``f`` — loss amplitude (QSHE only), subject to ``t² + f² ≤ 1``.
- ``ν``  — critical exponent extracted from EXP-run scaling; see ``calculate_nu``.
"""
# Some mathematical expressions in this file are long by necessity; disable
# the line-length rule for this file to keep the formulas readable.
# flake8: noqa: E501

import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
import json

from datetime import datetime, timezone

T_DICT = {"0": "random", "1": 0.0, "2": 0.5, "3": float(1 / np.sqrt(2)), "4": 1.0}
PHI_DICT = {
    "0": "random",
    "1": 0.0,
    "2": float(np.pi / 4),
    "3": float(np.pi / 2),
    "4": float(np.pi),
    "5": float(np.pi * 2),
}
THETA_DICT = {
    "0": "random",
    "1": 0.0,
    "2": float(np.pi / 8),
    "3": float(3 * np.pi / 16),
    "4": float(np.pi / 4),
    "5": float(3 * np.pi / 8),
    "6": float(np.pi / 2),
    "7": float(0.1),
    "8": float(7 * np.pi / 32),
}


# ---------- Misc. utility ---------- #
def save_data(
    hist_vals: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray, filename: str
) -> None:
    """Save histogram arrays to a compressed .npz file.

    Writes the three standard histogram arrays under the fixed keys
    ``histval``, ``binedges``, and ``bincenters`` that the rest of the
    pipeline expects when loading ``.npz`` histogram files.

    Args:
        hist_vals: Histogram counts (or densities) stored as ``histval`` in
            the archive.
        bin_edges: Bin edge array of length ``n_bins + 1``, stored as
            ``binedges``.
        bin_centers: Bin centre array of length ``n_bins``, stored as
            ``bincenters``.
        filename: Destination path; numpy appends ``.npz`` if not present.
    """
    np.savez_compressed(
        filename,
        histval=hist_vals,
        binedges=bin_edges,
        bincenters=bin_centers,
    )


def get_current_date(format: str = "full") -> str:
    """Return the current UTC date/time as a formatted string.

    Args:
        format: Resolution of the timestamp.  One of ``"day"``
            (``YYYY-MM-DD``), ``"hour"`` (``YYYY-MM-DD HH``),
            ``"min"`` (``YYYY-MM-DD HH:MM``), or ``"full"``
            (``YYYY-MM-DD HH:MM:SS``, default).

    Returns:
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


def save_metric_json(data: dict, filename: str):
    """Serialise a dictionary to a JSON file with 2-space indentation.

    Args:
        data: Dictionary to serialise.  Values must be JSON-serialisable;
            use ``collapse_data`` first if the dict contains numpy arrays.
        filename: Destination file path (created or overwritten).
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def collapse_data(data_dict: dict):
    """Recursively convert a nested structure into a JSON-serialisable form.

    Traverses the input recursively: dictionaries are rebuilt with converted
    keys and values; numpy arrays are converted to Python lists via
    ``.tolist()``; all other objects are returned unchanged.

    Args:
        data_dict: Nested dictionary (possibly containing numpy arrays) to
            convert.

    Returns:
        A JSON-serialisable copy of ``data_dict`` with numpy arrays replaced
        by plain Python lists.
    """
    if isinstance(data_dict, dict):
        return {
            collapse_data(key): collapse_data(value) for key, value in data_dict.items()
        }
    elif isinstance(data_dict, np.ndarray):
        return data_dict.tolist()
    else:
        return data_dict


# Potentially unused in active pipeline — verify against test_qshe.ipynb
# before removing.
def build_state_dict(
    qvals: np.ndarray,
    gvals: np.ndarray,
    nsamples: int,
    steps: int,
    metric: str,
    fix: int = 1,
    seed: int = 1234,
):
    """Build a state dictionary recording sweep parameters and run metadata.

    Summarises the q and p (g) sweep grids together with the run
    configuration so that output files are self-describing.

    Args:
        qvals: 1-D array of q initial values used in the sweep.
        gvals: 1-D array of p (g) initial values used in the sweep.
        nsamples: Number of MC samples per trial.
        steps: Number of RG iterations per trial.
        metric: Aggregation statistic used (``"mean"``, ``"median"``,
            ``"std"``, or ``"all"``).
        fix: Whether q is held fixed during RG iteration (``1``) or evolved
            (``0``).  Default: ``1``.
        seed: RNG seed used for reproducibility.  Default: ``1234``.

    Returns:
        A nested dictionary with keys ``"q"``, ``"p"``, and ``"data"``,
        each containing summary statistics and configuration values for
        the respective axis or run settings.
    """
    state = {}
    num_qs = int(qvals.size)
    num_gs = int(gvals.size)
    min_q = float(min(qvals))
    max_q = float(max(qvals))
    min_g = float(min(gvals))
    max_g = float(max(gvals))
    trimmed_qs = [round(float(q), 3) for q in qvals]
    trimmed_gs = [round(float(g), 3) for g in gvals]
    state.update(
        {
            "q": {
                "Num": num_qs,
                "Max": round(max_q, 3),
                "Min": round(min_q, 3),
                "Init": trimmed_qs,
            }
        }
    )
    state.update(
        {
            "p": {
                "Num": num_gs,
                "Max": round(max_g, 3),
                "Min": round(min_g, 3),
                "Init": trimmed_gs,
            }
        }
    )
    state.update(
        {
            "data": {
                "type": metric,
                "samples": nsamples,
                "steps": steps,
                "fixed": fix,
                "seed": seed,
            }
        }
    )
    return state


# Potentially unused in active pipeline — verify against test_qshe.ipynb
# before removing.
def get_meds(
    step_a: int,
    step_b: int,
    data_dict: dict,
    qarray: np.ndarray,
    garray: np.ndarray,
    var: str = "p",
    fromjson: bool = False,
):
    """Extract per-q mean and median values for two RG steps.

    Iterates over all q values in ``qarray`` and collects the median and
    mean of the observable ``var`` at two specified RG steps, supporting
    both live data dicts and pre-loaded JSON structures.

    Args:
        step_a: Index of the first RG step to extract.
        step_b: Index of the second RG step to extract.
        data_dict: Nested data dictionary keyed by q value (and then by g
            value if ``fromjson=False``).
        qarray: 1-D array of q values to iterate over.
        garray: 1-D array of g (p) initial values to iterate over.
        var: Observable name to extract when ``fromjson=True``.
            Default: ``"p"``.
        fromjson: If ``True``, read from a pre-loaded JSON structure where
            medians/means are stored under string step keys.  If ``False``,
            read from live trial-output arrays.  Default: ``False``.

    Returns:
        A 4-tuple ``(medgs_a, medgs_b, meangs_a, meangs_b)`` where each
        element is a dict mapping q → np.ndarray of values across ``garray``
        for that step.
    """
    medgs_a = {}
    medgs_b = {}
    meangs_a = {}
    meangs_b = {}
    for q in qarray[:]:
        # Other data extraction method if we load the json data instead
        if fromjson:
            gs = data_dict[f"{q}"][var]
            step_a_meds = gs[f"{step_a}"]["Median"]
            step_b_meds = gs[f"{step_b}"]["Median"]
            step_a_means = gs[f"{step_a}"]["Mean"]
            step_b_means = gs[f"{step_b}"]["Mean"]
        else:
            # Otherwise, we want to generate arrays of the mean/median for all ginits, for two consecutive RG steps.
            gs = data_dict[q]
            step_a_meds = []
            step_b_meds = []
            step_a_means = []
            step_b_means = []
            for ginit in garray:
                meang_a, medg_a = gs[ginit][step_a][0]
                meang_b, medg_b = gs[ginit][step_b][0]
                step_a_meds.append(medg_a)
                step_b_meds.append(medg_b)
                step_a_means.append(meang_a)
                step_b_means.append(meang_b)

        medgs_a.update({q: np.array(step_a_meds)})
        medgs_b.update({q: np.array(step_b_meds)})
        meangs_a.update({q: np.array(step_a_means)})
        meangs_b.update({q: np.array(step_b_means)})
    return medgs_a, medgs_b, meangs_a, meangs_b


# ---------- Data generators ---------- #
def build_rng(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy PCG64 random generator.

    Args:
        seed: Integer seed.  Use the same value across runs to reproduce
            identical MC sequences.

    Returns:
        A ``numpy.random.Generator`` backed by the PCG64 bit generator.
    """
    return np.random.default_rng(seed=seed)


def generate_constant_array(N: int, value: float, M: int = 1) -> np.ndarray:
    """Generate a constant-valued float64 array.

    Args:
        N: Number of rows (samples).
        value: Fill value for every element.
        M: Number of columns.  When ``M == 1`` a 1-D array of shape
            ``(N,)`` is returned; otherwise a 2-D array of shape
            ``(N, M)`` is returned.  Default: ``1``.

    Returns:
        Float64 array filled with ``value``.
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
    """Generate uniformly distributed random phase angles for the RG step.

    Args:
        N: Number of phase sets (rows) to generate.
        rng: Random number generator used to draw uniform variates.
        i: Number of independent phase values per sample (columns).
            Use ``4`` for the analytic IQHE path, ``8`` for the numerical
            IQHE path, and ``16`` for the QSHE path.  Default: ``4``.

    Returns:
        Array of shape ``(N, i)`` with phases drawn uniformly from
        ``[0, 2π)``.
    """
    phi_sample = rng.uniform(0, 2 * np.pi, (N, i))
    return phi_sample


def generate_initial_t_distribution(
    N: int, rng: np.random.Generator, upper_bound: float = 1.0
) -> np.ndarray:
    """Generate an initial flat distribution of transmission amplitudes.

    Draws squared amplitudes ``g ~ U[0, upper_bound]`` and returns
    ``t = √g``.  With ``upper_bound=1`` this produces a distribution
    symmetric about ``t² = 0.5``, suitable as an unbiased starting point.

    Args:
        N: Number of amplitude samples to generate.
        rng: Random number generator used to draw uniform variates.
        upper_bound: Upper bound for the uniform draw of ``g``.
            Useful when the QSHE unitarity constraint limits the
            available range.  Default: ``1.0``.

    Returns:
        1-D array of ``N`` amplitude values ``t ∈ [0, √upper_bound]``.
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
    """Generate the initial data arrays for a QSHE RG run.

    Constructs amplitude (``t``), loss (``f``), and phase (``phi``) arrays
    for ``samples`` Monte Carlo particles.  The ``t_val`` and ``phi_val``
    arguments select either a fixed value or a random draw via the
    ``T_DICT`` / ``PHI_DICT`` lookup tables; ``f_val == 0`` selects a fixed
    loss of zero.

    Args:
        samples: Number of MC samples to initialise.
        t_val: Index into ``T_DICT`` selecting the initial t distribution.
            ``0`` → random flat distribution; ``1–4`` → fixed values.
        phi_val: Index into ``PHI_DICT`` selecting the initial phase.
            ``0`` → random uniform; ``1–5`` → fixed values.
        f_val: Initial loss amplitude f ∈ [0, 1].  Clamped to [0, 1].
        rng: Random number generator used for random draws.

    Returns:
        Dictionary with keys:

        - ``"t"``    : float array of shape ``(samples, 5)`` — amplitudes.
        - ``"f"``    : float array of shape ``(samples, 5)`` — loss amplitudes.
        - ``"phi"``  : float array of shape ``(samples, 16)`` — phases.
        - ``"split"``: float array of shape ``(samples, 5)`` — available
          amplitude budget ``1 − f²`` per particle.
    """
    n = samples
    if f_val > 1.0:
        f_val = 1.0
    elif f_val < 0.0 or f_val < 1e-10:
        f_val = 0.0
    f_array = generate_constant_array(n, f_val, 5)
    if t_val == 0:
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
    """Draw a ``(N, 5)`` matrix of amplitude samples by random index selection.

    Each RG transformation step requires five independent amplitude samples
    per particle.  This function constructs that matrix by sampling with
    replacement from the current P(t) distribution.

    Args:
        t: 1-D array of ``N`` amplitude values representing the current
            P(t) distribution.  Values should be in ``[0, 1]``.
        N: Number of output rows (particles).
        rng: Random number generator used to draw integer indices.

    Returns:
        Array of shape ``(N, 5)`` where each row contains five amplitude
        values drawn independently from ``t``.
    """
    # Fancy-index t with a (N, 5) integer array: draws 5 random amplitude
    # samples per row by sampling with replacement from the N-element t array.
    t_sample = t[rng.integers(0, N, size=(N, 5))]
    return t_sample


# ---------- t prime computation ---------- #
def solve_matrix_eq(
    ts: np.ndarray, phis: np.ndarray, batch_size: int = 100000, output_index: int = 8
) -> np.ndarray:
    """Solve the 10×10 batched linear system Ax = b for the IQHE numerical RG step.

    Constructs a complex 10×10 scattering matrix A for each sample in the
    batch using the five transmission amplitudes and eight phase angles, then
    solves Ax = b simultaneously for the whole batch via
    ``numpy.linalg.solve``.  Returns the solution component at
    ``output_index``, whose magnitude gives t'.

    Notes:
        Solves the 10×10 complex scattering system for the 5-site IQHE
        network model (numerical 8-phase variant).

    Args:
        ts: Array of shape ``(batch_size, 5)`` — one row of five transmission
            amplitudes ``t1…t5 ∈ [0, 1]`` per sample.
        phis: Array of shape ``(batch_size, 8)`` — the eight inter-site phases
            used to populate the off-diagonal elements of A.
        batch_size: Number of samples in this batch.  Default: ``100000``.
        output_index: Component of the solution vector x to return.
            Index ``8`` corresponds to the outgoing complex amplitude whose
            magnitude is t'.  Default: ``8``.

    Returns:
        Array of shape ``(batch_size, 1)`` containing the complex solution
        component at ``output_index`` for each sample in the batch.
    """
    # Column-wise unpack: ts has shape (batch_size, 5); .T gives (5, batch_size),
    # so each variable is a 1-D array of length batch_size.
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

    # A has shape (batch_size, 10, 10): A[:, i, j] assigns element (i, j)
    # across all batch_size matrices simultaneously (broadcast over axis 0).
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
    output_indexes: list,
    inputs: ArrayLike,
) -> dict:
    """Solve the 20×20 batched scattering system Mx = b for the QSHE model.

    Extends the IQHE matrix formulation to include the loss amplitude f and
    spin-resolved channels, resulting in a 20×20 complex linear system.
    Solves the full batch simultaneously and returns selected solution
    components.

    Notes:
        Solves the 20×20 complex scattering system for the 5-site QSHE
        network model including loss channels.

    Args:
        ts: Array of shape ``(batch_size, 5)`` — transmission amplitudes
            ``t1…t5 ∈ [0, 1]``.
        fs: Array of shape ``(batch_size, 5)`` — loss amplitudes
            ``f1…f5``, subject to ``t_i² + f_i² ≤ 1``.
        phis: Array of shape ``(batch_size, 16)`` — the sixteen inter-site
            phases used to build M.
        batch_size: Number of samples in this batch.
        output_indexes: List of solution-vector indices to return (e.g.
            ``[2, 9, 10, 17]`` for the standard QSHE observables).
        inputs: 1-D array-like of four boundary condition amplitudes used
            to populate the b vector.

    Returns:
        Dictionary mapping each index in ``output_indexes`` to a complex
        array of shape ``(batch_size, 1)`` containing the corresponding
        solution component.
    """
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
    sol = {}
    # Outputs are index 2, 9, 10 and 17 in order of O3_up, O10_up, O1_down and O8_down
    # return x
    for output in output_indexes:
        sol.update({output: x[:, output]})
    return sol


def generate_t_prime(
    t: np.ndarray, phi: np.ndarray, expression: str = "shaw"
) -> np.ndarray:
    """Apply an analytic RG map to compute next-step transmission amplitudes t'.

    Implements the core analytic RG transformation for the IQHE model.
    Each sample row supplies five amplitudes (t1…t5) and four phases
    (φ1…φ4); their complementary amplitudes r_i = √(1 − t_i²) are derived
    internally.  The chosen formula variant computes a complex ratio
    (numerator / denominator) and returns its absolute value.

    Notes:
        Applies one of four analytic RG map variants: ``"jack"``, ``"cain"``,
        ``"shaw"`` (Shaw 2023 thesis, default), or ``"t"`` (Shaw Eq 2.13).

    Args:
        t: Array of shape ``(N, 5)`` — five transmission amplitudes per row,
            values in ``[0, 1]``.
        phi: Array of shape ``(N, 4)`` — four random phases per row,
            values in ``[0, 2π)``.
        expression: Which analytic formula to use.  First character is
            matched: ``"j"`` → jack, ``"c"`` → cain, ``"s"`` → shaw,
            ``"t"`` → shaw second matrix.  Default: ``"shaw"``.

    Returns:
        1-D array of ``N`` transformed amplitudes t' ∈ ``[0, 1)``.
        Values are not explicitly clipped here; the caller (``rg_data_workflow``)
        is responsible for clipping if required.

    Raises:
        ValueError: If ``expression`` does not start with a recognised
            character (``j``, ``c``, ``s``, or ``t``).
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
    """Compute t' for all N samples by batching calls to ``solve_matrix_eq``.

    Divides the ``N`` samples into chunks of ``batch_size`` and solves the
    10×10 IQHE linear system for each chunk, accumulating results into a
    pre-allocated output array.  ``N`` must be exactly divisible by
    ``batch_size`` (ensured by the caller).

    Args:
        ts: Array of shape ``(N, 5)`` — transmission amplitudes.
        phis: Array of shape ``(N, 8)`` — phase values for matrix construction.
        N: Total number of samples.
        batch_size: Number of samples per matrix-solve call.  Default:
            ``100000``.

    Returns:
        Array of shape ``(N, 1)`` containing ``|x[output_index]|`` for each
        sample — the numerically computed t'.
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
    """Dispatch the RG transformation to the analytic or numerical path.

    Routes the t' computation based on the first character of ``method``:
    analytic (``'a'``) calls ``generate_t_prime``; numerical (``'n'``) calls
    ``numerical_t_prime``.

    Args:
        method: Method selector string.  First character ``'a'`` → analytic
            closed-form; ``'n'`` → numerical batched matrix solve.
        ts: Input amplitude array of shape ``(N, 5)``.
        phis: Input phase array; shape ``(N, 4)`` for analytic or ``(N, 8)``
            for numerical.
        N: Total number of samples.
        expr: Expression identifier forwarded to ``generate_t_prime`` (e.g.
            ``"shaw"``, ``"jack"``, ``"cain"``).  Unused for numerical path.
        batch_size: Batch size for the numerical path.  Default: ``100000``.

    Returns:
        Array of t' values; shape ``(N,)`` for the analytic path or
        ``(N, 1)`` for the numerical path.

    Raises:
        ValueError: If ``method`` does not start with ``'a'`` or ``'n'``.
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
    output_indexes: list,
    inputs: ArrayLike,
    batch_size: int,
) -> dict:
    """Solve the QSHE 20×20 scattering system for all N samples via batching.

    Caps ``batch_size`` at ``N`` when ``N < batch_size``, then iterates
    over batches and accumulates absolute solution components into
    pre-allocated float64 arrays.

    Args:
        ts: Array of shape ``(N, 5)`` — transmission amplitudes.
        fs: Array of shape ``(N, 5)`` — loss amplitudes.
        phis: Array of shape ``(N, 16)`` — inter-site phases.
        N: Total number of samples.
        output_indexes: List of solution-vector indices to collect (e.g.
            ``[2, 9, 10, 17]``).
        inputs: Boundary condition amplitudes passed to ``solve_qshe_matrix``.
        batch_size: Maximum samples per batch.

    Returns:
        Dictionary mapping each index in ``output_indexes`` to a float64
        array of shape ``(N, 1)`` containing ``|x[index]|`` for all samples.
    """
    if batch_size > N:
        batch_size = N
    num_batches = N // batch_size
    outputs = {index: np.empty(shape=(N, 1), dtype=np.float64) for index in output_indexes}
    for i in range(num_batches):
        indexes = slice(i * batch_size, (i + 1) * batch_size)
        output_dict = solve_qshe_matrix(
            ts[indexes],
            fs[indexes],
            phis[indexes],
            batch_size,
            output_indexes,
            inputs,
        )
        for index in output_indexes:
            outputs[index][indexes] = np.abs(output_dict[index])
    return outputs


def qp_trials(
    q: np.ndarray | float,
    pval: np.ndarray | float,
    nsamples: int,
    nsteps: int,
    phis: np.ndarray,
    rng: np.random.Generator,
    metric: str = "all",
    fixed: int = 1,
    output_vars: list = ["t", "f"],
    input_vals: list = [1.0, 0.0, 0.0, 0.0],
    batch_size: int = 10000,
) -> tuple:
    """Run QSHE RG iterations for a single (q_init, p_init) starting point.

    Initialises ``nsamples`` particles near ``(q, pval)``, then iterates
    the QSHE numerical RG transformation for ``nsteps`` steps, recording
    summary statistics of p and q at each step.

    Args:
        q: Initial q value (spin-mixing parameter); held fixed when
            ``fixed == 1``, or evolved when ``fixed == 0``.
        pval: Initial p value (``p = |t|²``), used to seed the narrow
            amplitude distribution.
        nsamples: Number of MC particles per trial.
        nsteps: Number of RG iterations to perform.
        phis: Pre-generated phase array of shape ``(nsamples, 16)`` used
            in the QSHE matrix solve.
        rng: Random number generator.
        metric: Which statistic to record per step.  ``"mean"`` → mean only;
            ``"median"`` → median only; ``"std"`` → std only;
            ``"all"`` → (mean, median, std).  Default: ``"all"``.
        fixed: If ``1``, q is held fixed at its initial value throughout.
            If ``0``, q is evolved via the RG transformation.  Default: ``1``.
        output_vars: List of observable names to solve for; mapped to
            solution-vector indices via ``{"t": 2, "r": 9, "tau": 10,
            "f": 17}``.  Default: ``["t", "f"]``.
        input_vals: Boundary condition amplitudes forwarded to
            ``qshe_numerical_solver``.  Default: ``[1.0, 0.0, 0.0, 0.0]``.
        batch_size: Samples per batch in the QSHE solver.  Default:
            ``10000``.

    Returns:
        A 2-tuple ``(pmets, qmets)`` where each element is a float64 array
        of shape ``(nsteps, metdim)``.  ``metdim`` is ``1`` for single-metric
        modes or ``3`` for ``"all"`` (mean, median, std).
    """
    if metric != "all":
        metdim = 1
    else:
        metdim = 3
    pmets = np.empty(shape=(nsteps, metdim))
    qmets = np.empty(shape=(nsteps, metdim))
    var_index_map = {"t": 2, "r": 9, "tau": 10, "f": 17}
    output_indexes = [var_index_map[var] for var in output_vars]

    t_init = rng.uniform(np.sqrt(pval - 1e-6), np.sqrt(pval + 1e-6), nsamples)
    f_init = np.sqrt(q * (1 - t_init**2))
    # print(f"Performing {nsteps} RG iterations for initial q = {q}, p = {pval}")
    # print(f"Means: t = {np.mean(t_init)}, f = {np.mean(f_init)}")
    indices = rng.integers(0, nsamples, (nsamples, 5))
    for step in range(nsteps):
        ts = np.take(t_init, indices)
        fs = np.take(f_init, indices)
        outs = qshe_numerical_solver(
            ts, fs, phis, nsamples, output_indexes, input_vals, batch_size
        )
        # print(outs.keys())
        tp = outs[2]
        tp = np.clip(tp, 1e-9, 1 - 1e-9)
        p = tp**2
        if fixed == 1:
            qmed = q
            qmean = q
            f2 = (1 - p) * q
            fp = np.sqrt(f2)
            qstd = 0.0
        else:
            fp = outs[17]
            f2 = fp**2
            qprime = f2 / (1 - p)
            qmed = np.median(qprime)
            qmean = np.mean(qprime)
            qstd = np.std(qprime)
        # g = p + f2
        fp = np.clip(fp, 1e-9, 1 - 1e-9)
        pmed = np.median(p)
        pmean = np.mean(p)
        pstd = np.std(p)
        if metric == "median":
            pmets[step, 0] = pmed
            qmets[step, 0] = qmed
        elif metric == "mean":
            pmets[step, 0] = pmean
            qmets[step, 0] = qmean
        elif metric == "std":
            pmets[step, 0] = pstd
            qmets[step, 0] = qstd
        else:
            pmets[step, 0] = pmean
            pmets[step, 1] = pmed
            pmets[step, 2] = pstd
            qmets[step, 0] = qmean
            qmets[step, 1] = qmed
            qmets[step, 2] = qstd

        t_init = tp
        f_init = fp

    return pmets, qmets


def run_qp_trials(
    qvals: np.ndarray,
    pvals: np.ndarray,
    nsamples: int,
    nsteps: int,
    rng: np.random.Generator,
    metric: str = "all",
    fixed: int = 1,
    output_vars: Optional[list] = None,
    input_vals: Optional[list] = None,
    batch_size: int = 10000,
) -> tuple:
    """Run QSHE q-p trials for all combinations in the (q, p) grid.

    Iterates over every ``(qvals[i], pvals[j])`` pair, calls ``qp_trials``
    for each, and stores the results in pre-allocated 4-D arrays.  A new
    random phase array is generated once per q value to improve statistical
    independence between q slices.

    Args:
        qvals: 1-D array of q initial values (grid axis 0).
        pvals: 1-D array of p initial values (grid axis 1).
        nsamples: Number of MC particles per trial.
        nsteps: Number of RG iterations per trial.
        rng: Random number generator.
        metric: Aggregation statistic; forwarded to ``qp_trials``.
            Default: ``"all"``.
        fixed: Whether to hold q fixed; forwarded to ``qp_trials``.
            Default: ``1``.
        output_vars: Observable names; forwarded to ``qp_trials``.
            Defaults to ``["t", "f"]`` if ``None``.
        input_vals: Boundary amplitudes; forwarded to ``qp_trials``.
            Defaults to ``[1.0, 0.0, 0.0, 0.0]`` if ``None``.
        batch_size: Samples per batch; forwarded to ``qp_trials``.
            Default: ``10000``.

    Returns:
        A 2-tuple ``(p_trial_data, q_trial_data)`` where each element is a
        float64 array of shape ``(len(qvals), len(pvals), nsteps, metdim)``.
        ``metdim`` is ``1`` for single-metric modes or ``3`` for ``"all"``.
    """
    if output_vars is None:
        output_vars = ["t", "f"]
    if input_vals is None:
        input_vals = [1.0, 0.0, 0.0, 0.0]
    if metric != "all":
        met_dim = 1
    else:
        met_dim = 3
    plen = pvals.size
    qlen = qvals.size
    p_trial_data = np.empty(shape=(qlen, plen, nsteps, met_dim), dtype=np.float64)
    q_trial_data = np.empty(shape=(qlen, plen, nsteps, met_dim), dtype=np.float64)
    for q in range(qlen):
        phis = generate_random_phases(nsamples, rng, 16)
        for p in range(plen):
            # Trial output is tuple of lists, with elements in the order of RG steps.
            a, b = qp_trials(
                qvals[q],
                pvals[p],
                nsamples,
                nsteps,
                phis,
                rng,
                metric,
                fixed,
                output_vars,
                input_vals,
                batch_size,
            )
            p_trial_data[q, p, :, :] = a
            q_trial_data[q, p, :, :] = b
        if q % 10 == 0:
            print(f"Trial for {q}th q value completed.")
    return p_trial_data, q_trial_data


# ---------- Variable conversion helpers ---------- #
def convert_t_to_g(t: np.ndarray) -> np.ndarray:
    """Convert transmission amplitude t to squared amplitude g = |t|².

    Args:
        t: Array of transmission amplitude values.  Accepts real or complex
            arrays; for real inputs this is simply t².

    Returns:
        Array of squared amplitudes g = |t|², same shape as input,
        values in ``[0, 1]``.
    """
    return t * t


def convert_g_to_z(g: np.ndarray) -> np.ndarray:
    """Convert squared amplitude g to RG flow parameter z = ln((1−g)/g).

    At the critical fixed point the z-distribution is symmetric about
    z = 0; ``g = 0.5`` maps to ``z = 0``.

    Args:
        g: Array of squared amplitude values.  Should be in ``(0, 1)``
            strictly; values at the boundary diverge logarithmically.

    Returns:
        Array of z values spanning ℝ, same shape as input.

    Notes:
        Callers are responsible for clipping g away from 0 and 1 before
        calling this function to avoid logarithmic divergences.
    """
    return np.log((1.0 - g) / g)


def convert_z_to_g(z: np.ndarray) -> np.ndarray:
    """Convert RG flow parameter z to squared amplitude g = 1/(1 + exp(z)).

    Inverse of ``convert_g_to_z``.

    Args:
        z: Array of z values (RG flow parameter), spanning ℝ.

    Returns:
        Array of squared amplitudes g ∈ ``(0, 1)``, same shape as input.
    """
    return 1.0 / (1.0 + np.exp(z))


def convert_z_to_t(z: np.ndarray) -> np.ndarray:
    """Convert RG flow parameter z directly to transmission amplitude t.

    Computes ``t = √(1 / (1 + exp(z)))``, combining the z → g and g → t
    steps.

    Args:
        z: Array of z values spanning ℝ.

    Returns:
        Array of transmission amplitudes t ∈ ``(0, 1)``, same shape as input.
    """
    return np.sqrt(1.0 / (1.0 + np.exp(z)))


def convert_t_to_z(t: np.ndarray) -> np.ndarray:
    """Convert transmission amplitude t directly to RG flow parameter z.

    Computes ``z = ln(1/t² − 1)``, the composition of t → g and g → z.

    Args:
        t: Array of transmission amplitude values in ``(0, 1)``; values at
            the boundary produce ±∞.

    Returns:
        Array of z values spanning ℝ, same shape as input.
    """
    return np.log((1.0 / (t**2.0)) - 1.0)


def convert_t_to_geff(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute the effective squared amplitude g_eff = |t|² + |f|² (QSHE).

    In the QSHE model the total transmitted intensity includes both the
    coherent amplitude t and the loss channel f.

    Args:
        t: Array of transmission amplitudes.
        f: Array of loss amplitudes, subject to ``|t|² + |f|² ≤ 1``.

    Returns:
        Array of effective squared amplitudes g_eff, same shape as input.
    """
    t2 = t * t
    f2 = f * f
    return t2 + f2


def convert_geff_to_t(g_eff: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Recover transmission amplitude t from g_eff and loss amplitude f (QSHE).

    Computes ``t = √(g_eff − |f|²)``, the inverse of the f-channel
    contribution to the effective squared amplitude.

    Args:
        g_eff: Array of effective squared amplitudes ``|t|² + |f|²``.
        f: Array of loss amplitudes whose squared contribution is subtracted.

    Returns:
        Array of transmission amplitudes t, same shape as input.
    """
    f2 = f**2
    t2 = g_eff - f2
    return np.sqrt(t2)


def convert_zeff_to_t(z_eff: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Convert effective z parameter and loss to transmission amplitude t (QSHE).

    Computes ``t = √((1 − loss) / (1 + exp(z_eff)))``.  Both input arrays
    are ravelled to 1-D before arithmetic so the function handles
    column-vector inputs from the 2-D sampler naturally.

    Args:
        z_eff: Array of effective z values.  Ravelled to 1-D internally.
        loss: Array of loss values (|f|²) in ``[0, 1)``.  Ravelled to 1-D
            internally.

    Returns:
        1-D array of transmission amplitudes t.
    """
    t2 = (1 - loss.ravel()) / (1 + np.exp(z_eff.ravel()))
    return np.sqrt(t2)


def convert_z_to_x(z):
    """Convert RG flow parameter z to the alternative parametrisation x.

    Computes ``x = arcsinh(exp(z/2))``.

    Args:
        z: Array of z values spanning ℝ.

    Returns:
        Array of x values, same shape as input.
    """
    return np.arcsinh(np.exp(z / 2))


def convert_g_to_x(g, theta=0.0):
    """Convert squared amplitude g to x via the intermediate z parametrisation.

    Composes g → z → x: first computes ``z = ln((1−g)/g)``, then
    applies ``convert_z_to_x``.

    Args:
        g: Array of squared amplitude values in ``(0, 1)``.
        theta: Unused angular parameter (reserved for future use).

    Returns:
        Array of x values, same shape as input.
    """
    z = np.log((1 - g) / g)
    return convert_z_to_x(z)


def convert_x_to_z(x):
    """Convert alternative parametrisation x to RG flow parameter z.

    Computes ``z = ln(sinh²(x))``.  Inverse of ``convert_z_to_x``.

    Args:
        x: Array of x values.

    Returns:
        Array of z values, same shape as input.
    """
    return np.log(np.sinh(x) ** 2)


def convert_x_to_g(x):
    """Convert alternative parametrisation x to squared amplitude g.

    Composes x → z → g: applies ``convert_x_to_z`` then ``g = 1/(1+exp(z))``.

    Args:
        x: Array of x values.

    Returns:
        Array of squared amplitudes g ∈ ``(0, 1)``, same shape as input.
    """
    z = convert_x_to_z(x)
    g = 1 / (1 + np.exp(z))
    return g


# ---------- Sampling helpers decoupled from P_D ---------- #
def normalise_samplers(sampler: str) -> str:
    """Normalise a sampler name string to the internal short key.

    Accepts several common aliases for each sampling method and maps them
    to a single canonical key used internally.

    Args:
        sampler: Input sampler name.  Recognised aliases:
            inverse-CDF — ``"i"``, ``"inv"``, ``"cdf"``, ``"inverse"``;
            rejection — ``"r"``, ``"rej"``, ``"reject"``, ``"rejection"``.
            Matching is case-insensitive after stripping whitespace.

    Returns:
        ``"i"`` for inverse-CDF sampling or ``"r"`` for rejection sampling.

    Raises:
        ValueError: If ``sampler`` does not match any recognised alias.
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
    """Draw N continuous samples from a binned histogram distribution.

    Dispatches to either ``inverse_cdf_sampler`` or ``rejection_sampler``
    based on the normalised sampler key.

    Args:
        N: Number of samples to draw.
        hist_vals: Histogram bin counts.
        bin_edges: Bin edge array of length ``len(hist_vals) + 1``.
        bin_centers: Bin centre array (required by rejection sampler).
        rng: Random number generator.
        sampler_input: Sampler selector; passed through
            ``normalise_samplers``.  Default: ``"i"`` (inverse-CDF).

    Returns:
        1-D array of ``N`` continuous samples drawn from the histogram's
        implied probability distribution.

    Raises:
        KeyError: If the normalised sampler key is not ``"i"`` or ``"r"``.
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
    """Draw N samples from a binned histogram using the inverse-CDF method.

    Builds the empirical CDF from bin densities and widths, maps N uniform
    variates through the inverted CDF via binary search, then adds uniform
    jitter within each selected bin to produce continuous samples.

    Args:
        N: Number of samples to draw.
        hist_vals: Histogram counts per bin.
        bin_edges: Bin edge array of length ``len(hist_vals) + 1``.
        rng: Random number generator used to draw uniform variates.

    Returns:
        1-D array of ``N`` continuous samples drawn from the histogram's
        implied probability density function.
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
    """Draw N samples from a binned histogram using vectorised rejection sampling.

    Proposes batches of candidate (x, y) pairs uniformly within the
    histogram domain, accepts candidates where ``y ≤ normalised_density(x)``,
    and accumulates accepted samples until exactly N are collected.  Batch
    size adapts between ``min_batch_size`` (10 000) and
    ``max_batch_size`` (1 000 000) based on remaining count.

    Args:
        N: Number of samples to produce.
        hist_vals: Histogram counts per bin.
        bin_edges: Bin edge array of length ``len(hist_vals) + 1``.
        bin_centers: Bin centre array (accepted for API compatibility;
            not used directly in the algorithm).
        rng: Random number generator used to draw uniform variates.

    Returns:
        1-D array of exactly ``N`` continuous samples drawn from the
        histogram's implied probability density.
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
    vars: list,
    data1: np.ndarray,
    data2: np.ndarray,
    data1_bins: int,
    data2_bins: int,
    data1_range: tuple,
    data2_range: tuple,
    sym: bool = False,
) -> dict:
    """Construct a 2-D histogram and derive marginal densities.

    Builds a joint 2-D histogram for two variables, optionally symmetrises
    the first axis about its mode, and computes properly normalised 2-D and
    1-D probability densities.

    Args:
        vars: List of two variable name strings (e.g. ``["z", "f"]``) used
            to label the returned dictionary keys.
        data1: 1-D (or ravelled) array of samples for the first variable
            (histogram x-axis, labelled ``vars[0]``).
        data2: 1-D (or ravelled) array of samples for the second variable
            (histogram y-axis, labelled ``vars[1]``).
        data1_bins: Number of bins along the first axis.
        data2_bins: Number of bins along the second axis.
        data1_range: ``(min, max)`` range for the first axis.
        data2_range: ``(min, max)`` range for the second axis.
        sym: If ``True``, shift the first-axis distribution so its mode
            aligns with the grid centre (symmetrisation step).
            Default: ``False``.

    Returns:
        A dictionary with keys:

        - ``"{var1}_{var2}"``: ``{"histval": 2-D count array, "densities": 2-D density array}``
        - ``"{var1}"``: ``{"histval", "binedges", "bincenters", "densities"}`` — marginal along axis 0.
        - ``"{var2}"``: ``{"histval", "binedges", "bincenters", "densities"}`` — marginal along axis 1.
    """
    data1 = data1.ravel()
    data2 = data2.ravel()
    hist2d, z_edges, f_edges = np.histogram2d(
        data1,
        data2,
        bins=(data1_bins, data2_bins),
        range=(data1_range, data2_range),
        density=False,
    )
    var1 = vars[0]
    var2 = vars[1]
    print(f"Building 2D histogram for : {vars}")
    # Compute bin centers
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    f_centers = 0.5 * (f_edges[1:] + f_edges[:-1])

    # If we're symmetrising, manually symmetrise counts and densities
    if sym:
        mode = float(z_centers[np.argmax(hist2d.sum(axis=1))])
        output = np.empty_like(hist2d, dtype=float)
        new = z_centers
        old = z_centers - mode
        for j in range(hist2d.shape[1]):
            output[:, j] = np.interp(new, old, hist2d[:, j], left=0.0, right=0.0)
        print(f"Shifted zcenters by {mode}")
        hist2d = output

    # Compute probability densities
    # [:, None] makes dz a column vector (shape: n_z_bins, 1);
    # [None, :] makes df a row vector (shape: 1, n_f_bins).
    # Broadcasting produces a (n_z_bins, n_f_bins) area matrix.
    dz = np.diff(z_edges)[:, None]
    df = np.diff(f_edges)[None, :]
    area = dz * df

    # Compute 2D densities
    total = hist2d.sum()
    p_zf = hist2d / total / area

    # Obtain 1D densities and assert they are normalised
    # Multiply p_zf (n_z, n_f) by df (1, n_f) then sum over the f-axis to
    # marginalise over f and recover the z marginal density p(z).
    p_z = (p_zf * df).sum(axis=1)
    assert np.abs(np.sum(p_z * np.diff(z_edges)) - 1.0) <= 1e-12
    # Multiply p_zf (n_z, n_f) by dz (n_z, 1) then sum over the z-axis to
    # marginalise over z and recover the f marginal density p(f).
    p_f = (p_zf * dz).sum(axis=0)
    assert np.abs(np.sum(p_f * np.diff(f_edges)) - 1.0) <= 1e-12

    # Obtain 1D counts
    z_counts = hist2d.sum(axis=1)
    f_counts = hist2d.sum(axis=0)
    print(f"Hist keys are : {var1}_{var2}")
    # Store relevant data, labelled for intuitive access
    hist_data = {
        f"{var1}_{var2}": {"histval": hist2d, "densities": p_zf},
        var1: {
            "histval": z_counts,
            "binedges": z_edges,
            "bincenters": z_centers,
            "densities": p_z,
        },
        var2: {
            "histval": f_counts,
            "binedges": f_edges,
            "bincenters": f_centers,
            "densities": p_f,
        },
    }

    return hist_data


def rejection_sampler_2d(data_dict: dict, rng: np.random.Generator, N: int) -> tuple:
    """Draw N valid (z, f) pairs from a 2-D histogram, rejecting unitarity violations.

    Repeatedly calls ``inverse_cdf_2d`` in batches, then rejects any
    sample where ``t² + f² > 1`` (unitarity constraint), until exactly
    N accepted pairs are collected.

    Args:
        data_dict: Dictionary produced by ``build_2d_hist`` containing
            2-D histogram counts and marginal bin edges under keys
            ``"zf"``, ``"z"``, and ``"f"``.
        rng: Random number generator.
        N: Number of valid (z, f) pairs to produce.

    Returns:
        A 2-tuple ``(z_take, f_take)`` where each element is a float64
        array of length ``N`` containing accepted sample values.
    """
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
        g_eff_sample = convert_z_to_g(z_sample)
        t_sample = convert_geff_to_t(g_eff_sample, f_sample)
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
    """Draw N (z, f) pairs from a 2-D histogram using the inverse-CDF method.

    Flattens the 2-D count array to 1-D, constructs a CDF, inverts it
    with binary search to obtain flat 1-D indices, then unravels those
    indices back to (z_bin, f_bin) pairs.  Samples are placed uniformly
    within the selected rectangular cells.

    Args:
        data_dict: Dictionary with keys ``"zf"`` (2-D counts), ``"z"``
            (bin edges), and ``"f"`` (bin edges).
        rng: Random number generator.
        N: Number of pairs to draw.

    Returns:
        A 2-tuple ``(z_sample, f_sample)`` of float64 arrays of length
        ``N``.
    """
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
    # Unravel the flat CDF index back to 2-D (z_bin, f_bin):
    # integer division gives the row (z), modulo gives the column (f).
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
    data_dict: dict,
    rng: np.random.Generator,
    N: int,
    var2d: str,
) -> tuple:
    """Draw N valid (var1, var2) pairs using conditional 2-D histogram sampling.

    Samples the first variable (var1) from its marginal P(var1), then for
    each selected var1 bin samples var2 from the conditional P(var2 | var1)
    using the corresponding histogram row.  Empty rows fall back to the
    var2 marginal.  Rejects pairs where the unitarity constraint
    ``t² + mix ≤ 1`` is violated.

    Args:
        data_dict: Dictionary produced by ``build_2d_hist`` containing the
            2-D histogram and marginal arrays.  Keys are named after the
            two variables, e.g. ``"z_mix"``, ``"z"``, ``"mix"``.
        rng: Random number generator.
        N: Number of valid pairs to produce.
        var2d: Key for the joint histogram entry, formatted as
            ``"{var1}_{var2}"`` (e.g. ``"z_mix"``).

    Returns:
        A 2-tuple ``(z_take, mix_take)`` of float64 arrays of length ``N``
        containing accepted (var1, var2) sample values.
    """
    vars = var2d.split("_")
    var0 = vars[0]
    var1 = vars[1]
    # Load 2D counts and respective axis bins
    zmix_counts = data_dict[var2d]["histval"]
    z_edges = data_dict[var0]["binedges"]
    mix_edges = data_dict[var1]["binedges"]

    # total = zmix_counts.sum()

    # z_size, mix_size = zmix_counts.shape
    # Manually compute z marginal and construct the 1D z cdf
    z_marginal = zmix_counts.sum(axis=1)
    z_cdf = np.cumsum(z_marginal / z_marginal.sum())

    # Guard against floating point errors
    z_cdf[-1] = 1.0

    # Initialise output arrays
    z_take = np.empty(N, dtype=np.float64)
    mix_take = np.empty(N, dtype=np.float64)

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
        mix_bins = np.empty(batch_size, dtype=np.int64)

        # unique_z: sorted unique bin indices; inv_z: integer array the same
        # length as z_bins where inv_z[k] is the position of z_bins[k] in
        # unique_z, allowing vectorised per-bin conditional sampling below.
        unique_z, inv_z = np.unique(z_bins, return_inverse=True)

        # Loop until batch_size no. of f_bins is obtained
        for index, bin in enumerate(unique_z):
            # Check for similar z bins, and pull the f row for the corresponding bin
            similar = inv_z == index
            mix_row = zmix_counts[bin, :]
            mix_row_sum = mix_row.sum()

            # If the bin is empty, fallback to f_marginal
            if mix_row_sum <= 0:
                mix_row = zmix_counts.sum(axis=0)
                mix_row_sum = mix_row.sum()

            # For similar z bins, pull f bin indexes from the constructed f cdf
            mix_row_cdf = np.cumsum(mix_row / mix_row_sum)
            mix_row_cdf[-1] = 1.0
            mix_bins[similar] = np.searchsorted(
                mix_row_cdf, rng.random(similar.sum()), side="right"
            )

        # Define rectangle to sample within
        z_left = z_edges[z_bins]
        z_right = z_edges[z_bins + 1]
        mix_bottom = mix_edges[mix_bins]
        mix_top = mix_edges[mix_bins + 1]

        # Sample uniformly within rectangle
        z_diff = z_right - z_left
        mix_diff = mix_top - mix_bottom
        z_sample = z_left + z_diff * rng.random(size=batch_size)
        mix_sample = mix_bottom + mix_diff * rng.random(size=batch_size)
        t_sample = convert_zeff_to_t(z_sample, mix_sample)

        # Validity check
        mask = t_sample**2 + mix_sample <= 1.0 + 1e-12
        valid = mask.sum()
        if valid == 0:
            continue

        # Take valid indexes
        take = min(valid, remaining)
        z_take[filled : filled + take] = z_sample[mask][:take]
        mix_take[filled : filled + take] = mix_sample[mask][:take]
        filled += take

        if num_iters % 100 == 0:
            print(
                f"Rejection sampler iteration {num_iters}. {filled} samples accepted so far."
            )
    print(f"Took {num_iters} iterations in total.")
    return z_take, mix_take


def get_density(hist_vals: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Convert histogram bin counts into a normalised probability density.

    Divides counts by the total count and by each bin width so that the
    resulting array integrates to 1 over the bin support.

    Args:
        hist_vals: Bin counts array of length ``n_bins``.
        bin_edges: Bin edge array of length ``n_bins + 1``.

    Returns:
        Array of probability densities of the same length as ``hist_vals``,
        satisfying ``sum(density * bin_widths) = 1``.
    """
    bin_counts = hist_vals.astype(float)
    bin_widths = np.diff(bin_edges)
    total = np.sum(bin_counts)
    probabilities = bin_counts / (total * bin_widths)
    return probabilities


# ---------- Moments helpers ---------- #
def l2_distance(old_hist_val, new_hist_val, old_bins, new_bins) -> float:
    """Compute the L2 distance between two histogram distributions.

    Evaluates ``δ = √∫(Q_{k+1}(z) − Q_k(z))² dz`` by numerical
    integration over the shared bin grid.

    Args:
        old_hist_val: Bin counts for the reference (previous-step)
            histogram.
        new_hist_val: Bin counts for the new (current-step) histogram.
        old_bins: Bin edge array for the reference histogram.
        new_bins: Bin edge array for the new histogram (must be the same
            grid as ``old_bins``).

    Returns:
        Scalar L2 distance between the two normalised density functions.
    """
    # L2 distance between 2 histograms
    old_density = get_density(old_hist_val, old_bins)
    new_density = get_density(new_hist_val, new_bins)
    integrand = (new_density - old_density) ** 2
    dz = np.diff(old_bins)
    l2_distance = float(np.sqrt(np.sum(integrand * dz)))
    return l2_distance


def mean_squared_distance(old_hist_val, new_hist_val, old_bins, new_bins) -> float:
    """Compute Shaw's mean-squared distance (MSD) between two histograms.

    Evaluates ``MSD = mean_over_bins(√max(Q_{k+1}² − Q_k², 0))``.
    Negative differences (which can arise when a large shift is applied)
    are clipped to zero before taking the square root.

    Args:
        old_hist_val: Bin counts for the reference (previous-step)
            histogram.
        new_hist_val: Bin counts for the new (current-step) histogram.
        old_bins: Bin edge array for the reference histogram.
        new_bins: Bin edge array for the new histogram.

    Returns:
        Scalar Shaw MSD value; used as a convergence metric in RG runs.
    """
    # Shaw's MSD
    old_density = get_density(old_hist_val, old_bins)
    new_density = get_density(new_hist_val, new_bins)
    shaw_integrand = new_density**2 - old_density**2
    # Clip negative values that can occur when shifts are large
    shaw_integrand = np.clip(shaw_integrand, 0.0, None)
    return float(np.mean(np.sqrt(shaw_integrand)))


def hist_moments(hist_vals: np.ndarray, bins: np.ndarray) -> tuple:
    """Compute the mean and standard deviation of a binned distribution.

    Computes the first and second moments by numerical integration over
    the bin centres weighted by the normalised probability density.

    Args:
        hist_vals: Bin counts array of length ``n_bins``.
        bins: Bin edge array of length ``n_bins + 1``.

    Returns:
        A 2-tuple ``(mean, standard_deviation)`` as Python floats.
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
    """Symmetrise a binned z-histogram about z = 0 by averaging mirror bins.

    Enforces particle-hole symmetry by replacing each bin with the average
    of itself and its mirror-image bin on the opposite side of z = 0.

    Args:
        z_hist: 1-D array of histogram counts (or densities) over z bins,
            assumed to be arranged symmetrically around z = 0.
        z_bins: Accepted for call-site compatibility (source/helpers.py passes
            bin edges positionally) but never used inside this function.
            Do not remove without updating all call sites.

    Returns:
        Symmetrised histogram array of the same shape as ``z_hist``.
    """
    # z_hist[::-1] reverses the array so that bin k is paired with its
    # mirror bin -(k+1), enforcing symmetry about z = 0.
    symmetrised_z = 0.5 * (z_hist + z_hist[::-1])
    return symmetrised_z


# ---------- Nu calculator ---------- #
def calculate_nu(slope: float, rg_steps: int) -> float:
    """Compute the critical exponent ν from the EXP-run scaling slope.

    Uses the scaling relation ``ν = ln(2^rg_steps) / ln(|slope|)`` where
    ``slope`` is the gradient of the shifted z-peak position versus
    perturbation magnitude obtained from EXP runs.

    Args:
        slope: Gradient of the linear fit of z_peak vs perturbation shift
            (from EXP runs).  The absolute value is used.
        rg_steps: Number of RG iterations used in the EXP run; sets the
            scale factor ``2^rg_steps`` in the numerator.

    Returns:
        Critical exponent ν as a Python float.
    """
    nu = np.log(2**rg_steps) / np.log(np.abs(slope))

    return float(nu)
