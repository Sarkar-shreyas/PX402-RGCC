"""Global constants and environment-derived paths for the RG Monte Carlo pipeline.

Exports
-------
- Path variables (``data_dir``, ``local_dir``, ``root_dir``, ``taskfarm_dir``,
  ``qshe_dir``, ``config_file``, ``host``, ``remote_dir``, ``qshe_test_dir``)
  resolved from the ``.env`` file via ``python-dotenv``; downstream modules import
  these rather than hardcoding paths.
- Simulation constants: ``N``, ``SHIFTS``, ``CURRENT_VERSION``, ``NUM_RG``
- Plot-parameter dictionaries: ``LEGENDS``, ``YLIMS``, ``XLIMS``
- RG variable encoding maps: ``T_DICT``, ``PHI_DICT``, ``THETA_DICT``

Path resolution
---------------
``load_dotenv()`` populates ``os.environ`` from the local ``.env`` file.  All
path variables are then read from the environment, so the same code works on
every developer machine and on the HPC cluster without modification.  If any
required variable is absent the module raises ``RuntimeError`` immediately so
misconfigured environments are caught at import time rather than deep inside a
run.

Downstream usage
----------------
All other modules (e.g. :mod:`file_management`, :mod:`analysis.critical_exponent`)
import from this module instead of duplicating ``os.getenv`` calls or hardcoded
strings.
"""

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED = [
    "DATA_DIR",
    "LOCAL_DIR",
    "ROOT_DIR",
    "TASKFARM_DIR",
    "HOST",
    "REMOTE_DIR",
    "CONFIG_FILE",
]
missing_vars = [var for var in REQUIRED if var not in os.environ]
if missing_vars:
    raise RuntimeError(
        f" Missing required env vars: {missing_vars}.\n See README.md for setup instructions."
    )

# ---------------------------------------------------------------------------
# Filesystem paths — resolved from .env; see module docstring for details.
# ---------------------------------------------------------------------------
data_dir = os.getenv("DATA_DIR")
local_dir = os.getenv("LOCAL_DIR")
root_dir = os.getenv("ROOT_DIR")
taskfarm_dir = os.getenv("TASKFARM_DIR")
qshe_dir = os.getenv("QSHE_DIR")
config_file = os.getenv("CONFIG_FILE")
host = os.getenv("HOST")
remote_dir = os.getenv("REMOTE_DIR")
qshe_test_dir = os.getenv("QSHE_TEST_DIR")

# ---------------------------------------------------------------------------
# Monte Carlo sample count — total number of RG samples per run.
# Local test runs use 32 M; HPC production runs use 480 M (this value).
# ---------------------------------------------------------------------------
N = 480000000

# ---------------------------------------------------------------------------
# EXP perturbation shifts — small detunings away from the critical fixed point
# used in EXP runs to measure the growth rate of the relevant eigenvalue and
# hence extract the critical exponent ν.
# ---------------------------------------------------------------------------
SHIFTS = ["0.0", "0.003", "0.004", "0.005", "0.006"]

# ---------------------------------------------------------------------------
# Run identifier — labels the output directory for the current production run;
# encodes model type, method, and expression variant.
# ---------------------------------------------------------------------------
CURRENT_VERSION = "fp_iqhe_numerical_shaw"

# ---------------------------------------------------------------------------
# RG iteration depth — number of RG steps applied per run (HPC production).
# Local runs typically use 7; production runs use 9 for better convergence.
# ---------------------------------------------------------------------------
NUM_RG = 9

# ---------------------------------------------------------------------------
# Plot layout parameters — legend placement, axis limits for every observable.
# Keyed first by run type ("FP" / "EXP") then by observable name.
# ---------------------------------------------------------------------------

# Legend anchor positions for each observable in FP and EXP plots.
LEGENDS = {
    "FP": {
        "r": "upper left",
        "t": "upper left",
        "p": "upper left",
        "tau": "upper left",
        "f": "upper left",
        "mix": "upper left",
        "leak": "upper right",
        "surv": "upper right",
        "g": "upper right",
        "input_t": "upper left",
        "z": "upper left",
        "sym_z": "upper left",
    },
    "EXP": {
        "r": "upper left",
        "t": "upper left",
        "p": "upper left",
        "tau": "upper left",
        "f": "upper left",
        "mix": "upper left",
        "leak": "upper right",
        "surv": "upper right",
        "g": "upper right",
        "input_t": "upper left",
        "z": "upper left",
    },
}

# Y-axis display ranges; amplitude-like observables are bounded [0, 3],
# broader phase-space quantities extend to 5.  z uses a narrow probability
# density range [0, 0.3].
YLIMS = {
    "FP": {
        "r": (0.0, 3.0),
        "t": (0.0, 3.0),
        "p": (0.0, 3.0),
        "q": (0.0, 3.0),
        "tau": (0.0, 5.0),
        "f": (0.0, 5.0),
        "mix": (0.0, 5.0),
        "leak": (0.0, 3.0),
        "surv": (0.0, 5.0),
        "g": (0.0, 3.0),
        "input_t": (0.0, 3.0),
        "z": (0.0, 0.3),
        "sym_z": (0.0, 0.25),
    },
    "EXP": {
        "r": (0.0, 3.0),
        "t": (0.0, 3.0),
        "p": (0.0, 3.0),
        "tau": (0.0, 5.0),
        "f": (0.0, 5.0),
        "mix": (0.0, 5.0),
        "leak": (0.0, 3.0),
        "surv": (0.0, 5.0),
        "g": (0.0, 3.0),
        "input_t": (0.0, 3.0),
        "z": (0.0, 0.3),
    },
}

# X-axis display ranges; amplitude observables are plotted over [0, 1] (the
# physical range of t); z is plotted over [-25, 25] to capture the full
# log-ratio distribution.
XLIMS = {
    "FP": {
        "r": (0.0, 1.0),
        "t": (0.0, 1.0),
        "p": (0.0, 1.0),
        "q": (0.0, 1.0),
        "tau": (0.0, 1.0),
        "f": (0.0, 1.0),
        "mix": (0.0, 1.0),
        "leak": (0.0, 1.0),
        "surv": (0.0, 1.0),
        "g": (0.0, 1.0),
        "input_t": (0.0, 1.0),
        "z": (-25.0, 25.0),
        "sym_z": (-25.0, 25.0),
    },
    "EXP": {
        "r": (0.0, 1.0),
        "t": (0.0, 1.0),
        "p": (0.0, 1.0),
        "tau": (0.0, 1.0),
        "f": (0.0, 1.0),
        "mix": (0.0, 1.0),
        "leak": (0.0, 1.0),
        "surv": (0.0, 1.0),
        "g": (0.0, 1.0),
        "input_t": (0.0, 1.0),
        "z": (-25.0, 25.0),
    },
}

# ---------------------------------------------------------------------------
# RG variable encoding maps — integer keys map CLI/config integer codes to
# physical parameter values used in the RG transformation.
# ---------------------------------------------------------------------------

# Transmission amplitude t: 0 = uniform random; 1–4 = specific fixed values
# including the critical point (1/√2) and the localised/delocalised limits.
T_DICT = {"0": "random", "1": 0.0, "2": 0.5, "3": float(1 / np.sqrt(2)), "4": 1.0}

# Scattering phase φ: 0 = random; 1–5 cover key fractions of π.
PHI_DICT = {
    "0": "random",
    "1": 0.0,
    "2": float(np.pi / 4),
    "3": float(np.pi / 2),
    "4": float(np.pi),
    "5": float(np.pi * 2),
}

# Mixing angle θ: 0 = random; 1–8 span the range [0, π/2] at physically
# motivated fractions, including the SU(2)-symmetric point (π/4).
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
