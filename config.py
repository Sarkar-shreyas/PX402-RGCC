import os
from dotenv import load_dotenv

load_dotenv()

N = int(os.getenv("NUMBER_OF_ITERATIONS", 1e8))
K = int(os.getenv("NUMBER_OF_RG_STEPS", 50))
BINS = int(os.getenv("NUMBER_OF_BINS", 1e3))
Z_RANGE = (float(os.getenv("Z_MIN", -10.0)), float(os.getenv("Z_MAX", 10.0)))
Z_PERTURBATION = float(os.getenv("Z_PERTURBATION", 5e-4))
DIST_TOLERANCE = float(os.getenv("DIST_TOLERANCE", 1e-3))
STD_TOLERANCE = float(os.getenv("STD_TOLERANCE", 5e-4))
T_RANGE = (float(os.getenv("T_MIN", 0.0)), float(os.getenv("T_MAX", 1.0)))
