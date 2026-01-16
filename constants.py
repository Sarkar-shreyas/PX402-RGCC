"""Project-wide constants and environment-derived paths.

This module centralises constants used across the repo and reads required
environment variables from a `.env` file via `python-dotenv`. It is imported
by runtime helpers (for example, :mod:`file_management`) that rely on the
environment for remote host and path configuration.

Environment variables (required)
-------------------------------
- ``DATA_DIR``: Local destination root for pulled job outputs.
- ``LOCAL_DIR``: Local folder for locally produced data.
- ``ROOT_DIR``: Local repository root.
- ``TASKFARM_DIR``: Local path to the Taskfarm folder used for staging.
- ``HOST``: Remote SSH host used for transfers.
- ``REMOTE_DIR``: Remote project base directory on the cluster (referred to
  throughout the docs as ``<REMOTE_ROOT>``).

The module will raise ``RuntimeError`` at import time when any required env
vars are missing. This behaviour is intentional to fail fast when running
transfer utilities or local tests that expect these values to be available.
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

data_dir = os.getenv("DATA_DIR")
local_dir = os.getenv("LOCAL_DIR")
root_dir = os.getenv("ROOT_DIR")
taskfarm_dir = os.getenv("TASKFARM_DIR")
qshe_dir = os.getenv("QSHE_DIR")
config_file = os.getenv("CONFIG_FILE")
host = os.getenv("HOST")
remote_dir = os.getenv("REMOTE_DIR")


# SHIFTS = [0.0, 0.003, 0.005, 0.007, 0.009]
# SHIFTS = ["0.0", "0.003", "0.005", "0.0075", "0.010"]
# SHIFTS = ["0.0", "0.003", "0.004", "0.005", "0.006", "0.0075", "0.010"]
N = 480000000
SHIFTS = ["0.0", "0.003", "0.004", "0.005", "0.006"]
# CURRENT_VERSION = "1.84J"
# CURRENT_VERSION = "1.90S"
CURRENT_VERSION = "fp_iqhe_numerical_shaw"
NUM_RG = 9
# NUM_RG = 10
# NUM_RG = 12
LEGENDS = {
    "FP": {
        "t": "upper left",
        "g": "upper right",
        "input_t": "upper left",
        "z": "upper left",
        "sym_z": "upper left",
    },
    "EXP": {
        "t": "upper left",
        "g": "upper right",
        "input_t": "upper left",
        "z": "upper left",
    },
}
YLIMS = {
    "FP": {
        "t": (0.0, 3.0),
        "g": (0.0, 3.0),
        "input_t": (0.0, 3.0),
        "z": (0.0, 0.3),
        "sym_z": (0.0, 0.25),
    },
    "EXP": {
        "t": (0.0, 3.0),
        "g": (0.0, 3.0),
        "input_t": (0.0, 3.0),
        "z": (0.0, 0.3),
    },
}
XLIMS = {
    "FP": {
        "t": (0.0, 1.0),
        "g": (0.0, 1.0),
        "input_t": (0.0, 1.0),
        "z": (-25.0, 25.0),
        "sym_z": (-25.0, 25.0),
    },
    "EXP": {
        "t": (0.0, 1.0),
        "g": (0.0, 1.0),
        "input_t": (0.0, 1.0),
        "z": (-25.0, 25.0),
    },
}

T_DICT = {"0": "random", "1": 0.0, "2": 0.5, "3": float(1 / np.sqrt(2)), "4": 1.0}
PHI_DICT = {
    "0": "random",
    "1": 0.0,
    "2": float(np.pi / 4),
    "3": float(np.pi / 2),
    "4": float(np.pi),
    "5": float(np.pi * 2),
}
