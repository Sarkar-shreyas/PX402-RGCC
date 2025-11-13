#!/usr/bin/env python
"""
This file is in charge of taking in new data arrays and appending their histograms to the existing global histogram. The histograms are constructed using counts for ease of joining.
"""

import numpy as np

from .utilities import (
    T_BINS,
    T_RANGE,
    Z_BINS,
    Z_RANGE,
    save_data,
)
import sys


def construct_initial_histogram(
    data_file: str,
    output_filename: str,
    var: str,
) -> None:
    """A function to construct the initial histogram for any type of data"""
    data = np.load(data_file)
    if data.size == 0:
        raise FileNotFoundError(f"Could not load data from {data_file}")
    if var.lower() == "t" or var.lower() == "g":
        range = T_RANGE
        bins = T_BINS
    else:
        range = Z_RANGE
        bins = Z_BINS
        min_z, max_z = Z_RANGE
        data = data[np.isfinite(data)]
        z_mask = np.logical_and((data >= min_z), (data <= max_z))
        data = data[z_mask]

    hist_vals, bin_edges = np.histogram(data, bins, range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    save_data(
        hist_vals,
        bin_edges,
        bin_centers,
        output_filename,
    )


def append_to_histogram(
    input_file: str,
    existing_file,
    output_file: str,
    range: tuple,
) -> None:
    """A function to append the input data to an input histogram"""
    # Load the input data, should be a .npy file
    data = np.load(input_file)
    data = data[np.isfinite(data)]
    # Load the target file, should be an .npz file
    existing_data = np.load(existing_file, allow_pickle=False)
    existing_vals = existing_data["histval"]
    existing_bin_edges = existing_data["binedges"]
    existing_bin_centers = existing_data["bincenters"]
    # t_hist = t_data["histval"]
    # t_bin_edges = t_data["binedges"]
    # t_bin_centers = t_data["bincenters"]

    data_counts, _ = np.histogram(data, existing_bin_edges, range, density=False)
    assert data_counts.size == existing_vals.size
    existing_vals += data_counts
    save_data(existing_vals, existing_bin_edges, existing_bin_centers, output_file)


if __name__ == "__main__":
    if len(sys.argv) == 9:
        # This is if this is the first time we're making the histogram - no existing t or z histogram files
        process = int(sys.argv[1].strip())
        input_t_file = sys.argv[2].strip()
        input_g_file = sys.argv[3].strip()
        input_z_file = sys.argv[4].strip()
        output_t_file = sys.argv[5].strip()
        output_g_file = sys.argv[6].strip()
        output_z_file = sys.argv[7].strip()
        rg_step = int(sys.argv[8].strip())
    elif len(sys.argv) == 12:
        # If there are already existing histograms, we need to append the new data into them.
        process = int(sys.argv[1].strip())
        input_t_file = sys.argv[2].strip()
        input_g_file = sys.argv[3].strip()
        input_z_file = sys.argv[4].strip()
        existing_t_file = sys.argv[5].strip()
        existing_g_file = sys.argv[6].strip()
        existing_z_file = sys.argv[7].strip()
        output_t_file = sys.argv[8].strip()
        output_g_file = sys.argv[9].strip()
        output_z_file = sys.argv[10].strip()
        rg_step = int(sys.argv[11].strip())
    else:
        raise SystemExit(
            "Usage: histogram_manager.py PROCESS INPUT_T_FILE INPUT_G_FILE INPUT_Z_FILE EXISTING_T_FILE EXISTING_G_FILE EXISTING_Z_FILE OUTPUT_T_FILE OUTPUT_G_FILE OUTPUT_Z_FILE RG_STEP"
        )

    if process == 0:
        # This means we're going to be creating the first histograms of t and z
        print("-" * 100)
        print(f"Constructing initial histograms for RG step {rg_step}")
        construct_initial_histogram(input_t_file, output_t_file, "t")
        print(f"t histogram saved to {output_t_file}")
        construct_initial_histogram(input_g_file, output_g_file, "g")
        print(f"g histogram saved to {output_g_file}")
        construct_initial_histogram(input_z_file, output_z_file, "z")
        print(f"z histogram saved to {output_z_file}")
        # os.remove(input_t_file)
        # os.remove(input_g_file)
        # os.remove(input_z_file)
        print("-" * 100)
    elif process == 1:
        # This means we're just going to be appending the t data to the existing histograms
        print("-" * 100)
        print(f"Appending data to existing histograms for RG step {rg_step}")
        append_to_histogram(input_t_file, existing_t_file, output_t_file, T_RANGE)
        print(f"t histogram saved to {output_t_file}")
        append_to_histogram(input_g_file, existing_g_file, output_g_file, T_RANGE)
        print(f"g histogram saved to {output_g_file}")
        append_to_histogram(input_z_file, existing_z_file, output_z_file, Z_RANGE)
        print(f"z histogram saved to {output_z_file}")

        # Delete old files once done to prevent buildup
        # os.remove(input_t_file)
        # os.remove(input_g_file)
        # os.remove(input_z_file)
        # os.remove(existing_t_file)
        # os.remove(existing_g_file)
        # os.remove(existing_z_file)
        print("-" * 100)
    else:
        raise ValueError("Invalid Process: enter 0, 1, 2 or 3.")
