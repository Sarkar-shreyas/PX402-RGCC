#!/usr/bin/env python
"""
This is a helper file for making a histogram of input t data
"""

import numpy as np

from .utilities import (
    T_BINS,
    T_RANGE,
    Probability_Distribution,
    save_data,
)
import sys


def construct_initial_histogram(
    data_file: str,
    output_filename: str,
    bins: int,
    range: tuple,
    density: bool = False,
) -> None:
    """A function to construct the initial histogram for any type of data"""
    data = np.loadtxt(data_file)
    if data.size == 0:
        raise FileNotFoundError(f"Could not load data from {data_file}")

    distribution = Probability_Distribution(data, bins, range, density)
    save_data(
        distribution.histogram_values,
        distribution.bin_edges,
        distribution.bin_centers,
        output_filename,
    )


def append_to_histogram(
    input_file: str,
    existing_file,
    output_file: str,
    range: tuple,
) -> None:
    """A function to append the input data to an input histogram"""
    # Load the input data, should be a .txt file
    data = np.loadtxt(input_file)

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
    if len(sys.argv) == 6:
        # This is if this is the first time we're making the histogram - no existing t or z histogram files
        array_size = int(sys.argv[1].strip())
        process = int(sys.argv[2].strip())
        input_t_file = sys.argv[3].strip()
        output_t_file = sys.argv[4].strip()
        rg_step = int(sys.argv[5].strip())
    elif len(sys.argv) == 7:
        # If there are already existing histograms, we need to append the new data into them.
        array_size = int(sys.argv[1].strip())
        process = int(sys.argv[2].strip())
        input_t_file = sys.argv[3].strip()
        existing_t_file = sys.argv[4].strip()
        output_t_file = sys.argv[5].strip()
        rg_step = int(sys.argv[6].strip())
    else:
        raise SystemExit(
            "Usage: t_laundered_histogram_manager.py ARRAY_SIZE PROCESS INPUT_T_FILE EXISTING_T_FILE OUTPUT_T_FILE RG_STEP"
        )

    if process == 0:
        # This means we're going to be creating the first histograms of t and z
        print("-" * 100)
        print(f"Constructing initial input t histogram for RG step {rg_step}")
        construct_initial_histogram(input_t_file, output_t_file, T_BINS, T_RANGE, False)
        print(f"Input t histogram saved to {output_t_file}")
        # os.remove(input_t_file)
        # os.remove(input_g_file)
        # os.remove(input_z_file)
        print("-" * 100)
    elif process == 1:
        # This means we're just going to be appending the t data to the existing histograms
        print("-" * 100)
        print(f"Appending data to existing histogram for RG step {rg_step}")
        append_to_histogram(input_t_file, existing_t_file, output_t_file, T_RANGE)
        print(f"Input t histogram saved to {output_t_file}")

        # Delete old files once done to prevent buildup
        # os.remove(input_t_file)
        # os.remove(input_g_file)
        # os.remove(input_z_file)
        # os.remove(existing_t_file)
        # os.remove(existing_g_file)
        # os.remove(existing_z_file)
        print("-" * 100)
