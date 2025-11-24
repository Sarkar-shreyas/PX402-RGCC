"""Create shifted (perturbed) t-samples from a symmetrized Q(z) histogram.

This script loads a symmetrized z-histogram (saved as an `.npz` archive with
arrays `histval`, `binedges`, `bincenters`), generates `num_samples` via
the laundering routine, applies a constant shift (perturbation) to the z
samples, converts the shifted z-values to amplitudes `t`, and saves the
resulting t-array as a `.npy` file.

Typical usage:
    python shift_z.py NUM_SAMPLES INPUT_HIST.npz OUTPUT_T.npy SHIFT

Where `SHIFT` is a floating-point perturbation added to all z samples.
"""

import numpy as np
from .utilities import launder, convert_z_to_t
import sys

if __name__ == "__main__":
    num_samples = int(sys.argv[1].strip())
    input_file = sys.argv[2].strip()
    output_file = sys.argv[3].strip()
    shift = float(sys.argv[4].strip())

    perturbation = shift

    sym_z = np.load(input_file)
    sym_hist_vals = sym_z["histval"]
    sym_bins = sym_z["binedges"]
    sym_centers = sym_z["bincenters"]

    print("-" * 100)
    print(f"Loaded z histogram from {input_file}")

    sym_sample = launder(num_samples, sym_hist_vals, sym_bins, sym_centers)

    print(f"Laundered {num_samples} samples from loaded z histogram")

    shifted_sample = sym_sample + perturbation
    shifted_t = convert_z_to_t(shifted_sample)
    print(f"Shifted laundered sample by {perturbation}")
    np.save(output_file, shifted_t)

    print(f"Shifted t sample has been saved to {output_file}")

    print("-" * 100)
