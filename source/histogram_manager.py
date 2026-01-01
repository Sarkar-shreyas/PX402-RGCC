#!/usr/bin/env python
"""Build and append histograms for t and z variables.

This module contains helpers used by batch scripts to create initial
histograms from raw samples and to append new samples into an existing
histogram archive.
"""

import numpy as np

from source.utilities import (
    save_data,
)
from source.config import get_rg_config
import sys
from datetime import datetime, timezone


def _bin_and_range_manager(
    var: str, hist_vars: dict, shift: str | None = None
) -> tuple:
    """Select bin count and range for variable 't' or 'z'.

    Parameters
    ----------
    var : str
        Variable name, expected to be 't' or 'z' (case insensitive).
    shift : str or None
        Optional numeric shift (as string) to add to the z-range. If
        provided, the returned z-range will be shifted by this amount.

    Returns
    -------
    tuple
        (bins, range) where `bins` is an integer and `range` is a tuple
        (min, max) suitable for `numpy.histogram`.

    Raises
    ------
    ValueError
        If `var` is not one of 't' or 'z'.
    """
    if var != "t" and var != "z":
        raise ValueError(f"Invalid variable entered: {var}. Expected 't' or 'z'")
    bins = hist_vars[var]["bins"]
    range = hist_vars[var]["range"]
    if var == "z" and shift is not None:
        shift_val = float(shift.strip())
        min_z, max_z = range
        min_z += shift_val
        max_z += shift_val
        range = (min_z, max_z)
    return bins, range


def construct_initial_histogram(
    data: np.ndarray,
    output_filename: str,
    var: str,
    hist_vars: dict,
    shift: str | None = None,
) -> None:
    """Construct the initial histogram for the given data.

    Parameters
    ----------
    data : numpy.ndarray
        Raw data samples to histogram.
    output_filename : str
        Destination `.npz` filename for the saved histogram.
    var : str
        Variable name ('t' or 'z') to select binning parameters.
    shift : str or None
        Optional z-range shift to apply when computing bins.
    """
    # Drop nan values
    data = data[np.isfinite(data)]
    # Get bins and range for this variable and shift
    var = var.strip().lower()
    bins, range = _bin_and_range_manager(var, hist_vars, shift)

    hist_vals, bin_edges = np.histogram(data, bins=bins, range=range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    save_data(
        hist_vals,
        bin_edges,
        bin_centers,
        output_filename,
    )


def append_to_histogram(
    input_data: np.ndarray,
    existing_file: str,
    output_file: str,
    range: tuple,
) -> None:
    """Append input data to an existing histogram.

    Parameters
    ----------
    input_data : numpy.ndarray
        Raw input samples to append (will be histogrammed).
    existing_file : str
        Path to existing `.npz` histogram archive (must contain compatible
        bin edges).
    output_file : str
        Destination `.npz` filename for the updated histogram.
    """
    # Drop nan values
    input_data = input_data[np.isfinite(input_data)]

    # Load the target file, should be an .npz file
    existing_data = np.load(existing_file, allow_pickle=False)
    existing_vals = existing_data["histval"]
    existing_bin_edges = existing_data["binedges"]
    existing_bin_centers = existing_data["bincenters"]

    # Compute counts of input data. Bins should be the same so it doesn't need referencing.
    data_counts, _ = np.histogram(
        input_data, bins=existing_bin_edges, range=range, density=False
    )
    if data_counts.size != existing_vals.size:
        raise ValueError(
            f"Histogram sizes mismatched: Input: {data_counts.size}, Existing: {existing_vals.size}"
        )

    # Add counts of input data to the existing histogram
    existing_vals += data_counts
    save_data(existing_vals, existing_bin_edges, existing_bin_centers, output_file)


if __name__ == "__main__":
    input_length = len(sys.argv)
    if input_length not in [6, 7, 8]:
        raise SystemExit(
            " Usage: histogram_manager.py PROCESS VAR_NAME INPUT_FILE [EXISTING_FILE] OUTPUT_FILE RG_STEP [SHIFT] \n"
            " PROCESS 0 : Initialise histogram for input variable \n"
            " PROCESS 1 : Append input data to existing histogram "
        )
    process = int(sys.argv[1].strip())
    var_name = sys.argv[2].strip().lower()
    input_file = sys.argv[3].strip()
    rg_config = get_rg_config()
    hist_vars = {
        "z": {"bins": rg_config.z_bins, "range": rg_config.z_range},
        "t": {"bins": rg_config.t_bins, "range": rg_config.t_range},
    }
    if process == 0:
        # Then we're making the initial histogram, so there's no existing file input
        output_file = sys.argv[4].strip()
        rg_step = int(sys.argv[5].strip())
        mode = "Initialise"
        if input_length == 7:
            shift = sys.argv[6].strip()
        else:
            shift = None
    elif process == 1:
        # Then we're appending to an existing histogram
        existing_file = sys.argv[4].strip()
        output_file = sys.argv[5].strip()
        rg_step = int(sys.argv[6].strip())
        mode = "Append"
        if input_length == 8:
            shift = sys.argv[7].strip()
        else:
            shift = None
    else:
        raise SystemExit(
            "Invalid process entered. Process must be either 0 (Build new hist) or 1 (Append to existing hist)"
        )
    print("-" * 100)
    if shift is not None and len(shift) == 0:
        shift = None
    current_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{current_date}] : Histogram job of [RG step {rg_step}] with mode [{mode}] started for var {var_name} and shift {shift}"
    )
    data = np.load(input_file)
    if process == 0:
        construct_initial_histogram(data, output_file, var_name, hist_vars, shift)
        print(f"Histogram saved to {output_file}")
    else:
        if not existing_file:
            raise SystemExit(f"No existing histogram was found for mode {mode}")
        else:
            bins, range = _bin_and_range_manager(var_name, hist_vars, shift)
            append_to_histogram(data, existing_file, output_file, range)
            print(f"Appended input data to existing data at {existing_file}")

    print("-" * 100)
