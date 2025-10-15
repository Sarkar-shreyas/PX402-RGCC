import numpy as np


# ---------- Variable conversion helpers ---------- #
def convert_t_to_g(t: np.ndarray) -> np.ndarray:
    """Simple function returning g(t) = |t|^2"""
    return np.abs(t) * np.abs(t)


def convert_g_to_z(g: np.ndarray) -> np.ndarray:
    """Simple function returning z = ln((1-g)/g), clipping g values to prevent pooling at 0,1"""
    tolerance = 1e-15
    g = np.clip(g, tolerance, 1 - tolerance)
    return np.log((1.0 - g) / g)


def convert_z_to_g(z: np.ndarray) -> np.ndarray:
    """Simple function converting z to g for later use"""
    return 1.0 / (1.0 + np.exp(z))


def convert_t_to_z(t: np.ndarray) -> np.ndarray:
    """Simple function to convert t to z using earlier helpers"""
    return convert_g_to_z(convert_t_to_g(t))
