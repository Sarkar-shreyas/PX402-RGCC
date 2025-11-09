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
    np.savetxt(output_file, shifted_t)

    print(f"Shifted t sample has been saved to {output_file}")

    print("-" * 100)
