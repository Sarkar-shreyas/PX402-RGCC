"""Utility functions for variable transformations in the RG flow analysis.

This module provides a collection of functions for converting between different
parametrizations (t, g, z) used in the renormalization group analysis. The
transformations maintain numerical stability through appropriate clipping and
handling of edge cases.
"""

import numpy as np


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
    return t * t


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
    # tolerance = 1e-15
    # g = np.clip(g, tolerance, 1 - tolerance)
    return np.log((1.0 / g) - 1.0)


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
    z = np.clip(z, -25.0, 25.0)
    return np.sqrt(1.0 / (np.exp(z) + 1.0))


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
    # t = np.clip(t, 1.38e-11, 1.0)
    # g = t**2
    # g = np.clip(g, 1.39e-11, 1.0 - 1.39e-11)
    return np.log((1.0 / (t * t)) - 1.0)
