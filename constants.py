import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED = ["DATA_DIR", "LOCAL_DIR", "ROOT_DIR", "TASKFARM_DIR", "HOST", "REMOTE_DIR"]
missing_vars = [var for var in REQUIRED if var not in os.environ]
if missing_vars:
    raise RuntimeError(
        f" Missing required env vars: {missing_vars}.\n See README.md for setup instructions."
    )

data_dir = os.getenv("DATA_DIR")
local_dir = os.getenv("LOCAL_DIR")
root_dir = os.getenv("ROOT_DIR")
taskfarm_dir = os.getenv("TASKFARM_DIR")
config_file = os.getenv("CONFIG_FILE")
host = os.getenv("HOST", "vulcan2")
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

T_DICT = {"1": 0.0, "2": 0.5, "3": float(1 / np.sqrt(2)), "4": 1.0}
PHI_DICT = {
    "1": 0.0,
    "2": float(np.pi / 4),
    "3": float(np.pi / 2),
    "4": float(np.pi),
    "5": float(np.pi * 2),
}
