"""
This file tests which basis is the relevant scaling mode for the QSHE RG flow
"""

import numpy as np
from Local.run_local_qshe import (
    build_2d_hist,
    build_hist,
    single_qshe_rg_step,
    qshe_sampler,
)
from pathlib import Path

from source.config import RGConfig
from source.utilities import convert_geff_to_t, convert_z_to_g, convert_zeff_to_t


def load_2d_hist(hist2d_file: str | Path) -> dict:
    """Loads 2D hist data"""
    if not isinstance(hist2d_file, Path):
        filepath = Path(hist2d_file)
    else:
        filepath = hist2d_file

    data = np.load(filepath, allow_pickle=True)
    data2d = data["z_f"].item()
    data_z = data["z"].item()
    data_f = data["f"].item()

    zf_density = data2d["densities"]
    zf_counts = data2d["histval"]
    z_edges = data_z["binedges"]
    z_centers = data_z["bincenters"]
    f_edges = data_f["binedges"]
    f_centers = data_f["bincenters"]

    return {
        "z_f": {"histval": zf_counts, "densities": zf_density},
        "z": {
            "histval": data_z["histval"],
            "binedges": z_edges,
            "bincenters": z_centers,
            "densities": data_z["densities"],
        },
        "f": {
            "histval": data_f["histval"],
            "binedges": f_edges,
            "bincenters": f_centers,
            "densities": data_f["densities"],
        },
    }


def inner_prod(
    histA: np.ndarray, histB: np.ndarray, dz: np.ndarray, df: np.ndarray
) -> float:
    """Inner product of 2 histograms"""
    return np.sum(histA * histB * dz[:, None] * df[None, :])


def gram_schmidt_n(bases: list, dz: np.ndarray, df: np.ndarray) -> list:
    """Compute a set of orthonormal basis vectors for n modes via n-dimensional gram-schmidt"""
    ortho_vecs = []
    for basis in bases:
        vec = basis.copy()
        for ortho in ortho_vecs:
            vec -= inner_prod(ortho, vec, dz, df) * ortho
        norm = np.sqrt(inner_prod(vec, vec, dz, df))
        if norm < 1e-14:
            continue
        ortho_vecs.append(vec / norm)
    return ortho_vecs


def finite_diff(hist2d: np.ndarray, axis: int) -> np.ndarray:
    """Computes partial derivative of hist along input axis"""
    data = np.zeros_like(hist2d)

    # Central differences for interior, one-sided at edges
    if axis == 0:
        data[1:-1, :] = (hist2d[2:, :] - hist2d[:-2, :]) / 2.0
        data[0, :] = hist2d[1:, :] - hist2d[0, :]
        data[-1, :] = hist2d[-1, :] - hist2d[-2, :]
    elif axis == 1:
        data[:, 1:-1] = (hist2d[:, 2:] - hist2d[:, :-2]) / 2.0
        data[:, 0] = hist2d[:, 1:] - hist2d[:, 0]
        data[:, -1] = hist2d[:, -1] - hist2d[:, -2]
    else:
        raise ValueError(f"Axis must be 0 or 1. Got : {axis}")
    return data


def mean_array(hist2d: np.ndarray, dz: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Returns a constant array of integral(hist2d)/integral(1)"""
    integral = np.sum(hist2d * dz[:, None] * df[None, :])
    i = np.sum(dz[:, None] * df[None, :])
    means = integral / i
    return means


def weighted_mean(
    hist1: np.ndarray, hist2d: np.ndarray, dz: np.ndarray, df: np.ndarray
) -> np.ndarray:
    """Compute the weighted mean of 1 marginal from the 2d array"""
    numerator = np.sum(hist1 * hist2d * dz[:, None] * df[None, :])
    denominator = np.sum(hist2d * dz[:, None] * df[None, :])
    return numerator / denominator


def normalize_hist(hist2d: np.ndarray, dz: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Manually normalise 2D histogram densities"""
    integral = np.sum(hist2d * dz[:, None] * df[None, :])
    assert integral > 0.0
    return hist2d / integral


def basis_shift_gen(
    density2d: np.ndarray,
    z_centers: np.ndarray,
    f_centers: np.ndarray,
    dz: np.ndarray,
    df: np.ndarray,
) -> dict:
    """Compute bases for shift modes"""
    zc = z_centers[:, None]
    fc = f_centers[None, :]

    # Pure z and f shifts
    e_z = -finite_diff(density2d, 0)
    e_f = -finite_diff(density2d, 1)

    # Mean shifted modes
    e_z_mshift = (zc - weighted_mean(zc, density2d, dz, df)) * density2d
    e_f_mshift = (fc - weighted_mean(fc, density2d, dz, df)) * density2d

    # Correlation shift
    e_corr = (
        (zc - weighted_mean(zc, density2d, dz, df))
        * (fc - weighted_mean(fc, density2d, dz, df))
        * density2d
    )

    modes = [e_z, e_f, e_z_mshift, e_f_mshift, e_corr]
    # normalise all shifted datasets
    normed_modes = [m - mean_array(m, dz, df) for m in modes]

    # Get orthonormal basis vectors
    bases = gram_schmidt_n(normed_modes, dz, df)

    return {
        "e_z": bases[0],
        "e_f": bases[1],
        "e_z_mshift": bases[2],
        "e_f_mshift": bases[3],
        "e_corr": bases[4],
    }


def shift_density(
    hist2d: np.ndarray, basis: np.ndarray, delta: float, dz: np.ndarray, df: np.ndarray
) -> tuple:
    """Compute shifted density distributions and normalise"""
    new_dens = hist2d + delta * basis
    negatives = np.sum(np.clip(-new_dens, 0, None) * dz[:, None] * df[None, :])
    new_dens = np.clip(new_dens, 0, None)
    new_dens = normalize_hist(new_dens, dz, df)
    return new_dens, negatives


def rg_step(config: RGConfig, hist2d: dict, rng: np.random.Generator):
    samples = config.samples
    zsamp, lsamp = qshe_sampler(samples, rng, hist2d)
    ssamp = 1 - lsamp
    tsamp = convert_zeff_to_t(zsamp, lsamp)

    pass
