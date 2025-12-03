N = 480000000
DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm"
# CURRENT_VERSION = "2.00S"
CURRENT_VERSION = "1.90S"
# NUM_RG = 9
# NUM_RG = 10
NUM_RG = 12
# SHIFTS = [0.0, 0.003, 0.005, 0.007, 0.009]
# SHIFTS = ["0.0", "0.003", "0.005", "0.0075", "0.010"]
# SHIFTS = ["0.0", "0.003", "0.004", "0.005", "0.006", "0.0075", "0.010"]
SHIFTS = ["0.0", "0.003", "0.004", "0.005", "0.006"]

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
