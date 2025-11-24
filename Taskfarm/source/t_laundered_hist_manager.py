#!/usr/bin/env python
"""Helpers to construct and append histograms for laundered t data.

This module assists batch jobs that convert laundered t arrays into
histogram archives and append new t samples to existing archives.
"""

import numpy as np

from .utilities import (
    T_BINS,
    T_RANGE,
    save_data,
)
import sys


def construct_initial_histogram(
    data_file: str,
    output_filename: str,
    var: str,
) -> None:
    """Construct the initial histogram for laundered t data.

    Parameters
    ----------
    data_file : str
        Path to the `.npy` file containing laundered t samples.
    output_filename : str
        Destination `.npz` filename for the saved histogram.
    var : str
        Variable name; expected values: 't' or 'g'.
    """
    data = np.load(data_file)
    if data.size == 0:
        raise FileNotFoundError(f"Could not load data from {data_file}")
    if var.lower() == "t" or var.lower() == "g":
        range = T_RANGE
        bins = T_BINS

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
    """Append laundered t samples to an existing histogram archive.

    Parameters
    ----------
    input_file : str
        Path to the `.npy` file with laundered t samples.
    existing_file : str
        Path to the existing histogram `.npz` file.
    output_file : str
        Destination `.npz` filename for the updated histogram.
    range : tuple
        Range to use when computing counts (passed to numpy.histogram).
    """
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
    if len(sys.argv) == 5:
        # This is if this is the first time we're making the histogram - no existing t or z histogram files
        process = int(sys.argv[1].strip())
        input_t_file = sys.argv[2].strip()
        output_t_file = sys.argv[3].strip()
        rg_step = int(sys.argv[4].strip())
    elif len(sys.argv) == 6:
        # If there are already existing histograms, we need to append the new data into them.
        process = int(sys.argv[1].strip())
        input_t_file = sys.argv[2].strip()
        existing_t_file = sys.argv[3].strip()
        output_t_file = sys.argv[4].strip()
        rg_step = int(sys.argv[5].strip())
    else:
        raise SystemExit(
            "Usage: t_laundered_histogram_manager.py PROCESS INPUT_T_FILE EXISTING_T_FILE OUTPUT_T_FILE RG_STEP"
        )

    if process == 0:
        # This means we're going to be creating the first histograms of t and z
        print("-" * 100)
        print(f"Constructing initial input t histogram for RG step {rg_step}")
        construct_initial_histogram(input_t_file, output_t_file, "t")
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
