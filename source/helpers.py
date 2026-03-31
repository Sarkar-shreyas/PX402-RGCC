#!/usr/bin/env python
"""Convenience CLI helpers for laundering, symmetrising and converting data.

This script exposes a small set of command-line utilities used during the
data-management step of the RG workflow. It is intentionally thin and wraps
functionality from :mod:`source.utilities` to provide a stable
command-line interface for batch scripts and job arrays.

Supported PROCESS values (first CLI argument):

0 - Launder from z-histogram and convert to t
    Input:  NPZ z-histogram (histval, binedges, bincenters)
    Output: raw 1-D float64 ``.npy`` array of t amplitude samples
    Operation: draws ARRAY_SIZE samples via inverse-CDF or rejection
    sampling from the z-histogram, then maps each z value to a t
    amplitude via ``t = sqrt(1 / (1 + exp(z)))``.

1 - Symmetrise z-histogram
    Input:  NPZ z-histogram (histval, binedges, bincenters)
    Output: NPZ z-histogram with the same schema
    Operation: folds the z-distribution about zero — averages each bin
    with its mirror-image bin — to enforce the particle-hole symmetry
    expected at the RG fixed point.

2 - Launder from t-histogram
    Input:  NPZ t-histogram (histval, binedges, bincenters)
    Output: raw 1-D float64 ``.npy`` array of t amplitude samples
    Operation: draws ARRAY_SIZE samples directly from the t-histogram
    without a z→t conversion step (used for runs that track t only).

3 - Convert t array to z array
    Input:  raw 1-D float64 ``.npy`` array of t amplitude samples
    Output: raw 1-D float64 ``.npy`` array of z values
    Operation: applies ``z = ln((1 − t²) / t²)`` element-wise.

NPZ histogram schema
---------------------
All histogram NPZ files consumed and produced by this module share the
following array keys:

- ``histval``    — float64 array of shape ``(n_bins,)``: normalised histogram
  counts (probability density, not raw counts).
- ``binedges``   — float64 array of shape ``(n_bins + 1,)``: left/right edges
  of every bin.
- ``bincenters`` — float64 array of shape ``(n_bins,)``: midpoint of each bin,
  equal to ``0.5 * (binedges[:-1] + binedges[1:])``.

Load with ``np.load(file, allow_pickle=False)``.

The script is designed for use within higher-level shell job scripts that
orchestrate the RG pipeline. Errors raise SystemExit with a usage message so
they are visible in logs when a job fails.
"""

import numpy as np
import sys
from time import time
from datetime import datetime, timezone
from source.utilities import (
    save_data,
    launder,
    center_z_distribution,
    convert_z_to_t,
    convert_t_to_z,
    build_rng,
)
from source.config import get_rg_config

if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(
            " Usage: helpers.py PROCESS ARRAY_SIZE INPUT_FILE OUTPUT_FILE SEED \n"
            " PROCESS 0 : Launder from z-histogram + Convert to t \n"
            " PROCESS 1 : Symmetrise z-histogram \n"
            " PROCESS 2 : Launder from t-histogram \n"
            " PROCESS 3 : Convert t to z \n"
        )
    process = int(sys.argv[1].strip())
    array_size = int(sys.argv[2].strip())
    input_file = sys.argv[3].strip()
    output_file = sys.argv[4].strip()
    seed = int(sys.argv[5].strip())
    start = time()
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rng = build_rng(seed)
    rg_config = get_rg_config()
    sampler = rg_config.resample
    if process == 0:
        # Launder from input z histogram and convert to t data
        print("-" * 100)
        print(f"[{date}]: Began Laundering from z-histogram at {input_file}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        laundered_data = launder(
            array_size, input_hist, input_bin_edges, input_bin_centers, rng, sampler
        )
        laundered_t = convert_z_to_t(laundered_data)

        np.save(output_file, laundered_t)
        print(f"Laundering completed in {time() - start:.3f} seconds")
        print(f"Data in {input_file} laundered and saved to {output_file}")
        print("-" * 100)
    elif process == 1:
        # Symmetrise z histogram
        print("-" * 100)
        print(f"[{date}]: Symmetrising z-histogram in {input_file}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        symmetrized_data = center_z_distribution(input_hist, input_bin_edges)
        save_data(symmetrized_data, input_bin_edges, input_bin_centers, output_file)
        print(f"Symmetrisation completed in {time() - start:.3f} seconds")
        print(f"Symmetrised data saved to {output_file}")
        print("-" * 100)
    elif process == 2:
        # Launder from t-histogram for unsymmetrised runs
        print("-" * 100)
        print(f"[{date}]: Began Laundering from t-histogram at {input_file}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        laundered_t = launder(
            array_size, input_hist, input_bin_edges, input_bin_centers, rng, sampler
        )
        np.save(output_file, laundered_t)
        print(f"Laundering completed in {time() - start:.3f} seconds")
        print(f"Data in {input_file} laundered and saved to {output_file}")
        print("-" * 100)
    elif process == 3:
        # Convert an input t array into z data
        print("-" * 100)
        print(f"[{date}]: Converting t-data into z-data")
        t = np.load(input_file)
        z = convert_t_to_z(t)
        np.save(output_file, z)
        print(f"Conversion completed in {time() - start:.3f} seconds")
        print(f"t data in {input_file} converted and saved as z data to {output_file}")
        print("-" * 100)
    else:
        raise SystemExit("Invalid PROCESS entered. Valid: [0, 1, 2, 3]")
    # Delete old files once done to prevent buildup
    # os.remove(input_file)
