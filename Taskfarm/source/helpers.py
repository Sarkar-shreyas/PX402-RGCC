#!/usr/bin/env python
"""
This is a placeholder file meant to let me directly either apply laundering to an input array,
or symmetrise an input histogram for convenience
"""

import numpy as np
import sys
from time import time
from datetime import datetime, timezone
from .utilities import save_data, launder, center_z_distribution, convert_z_to_t
import os

if __name__ == "__main__":
    process = int(sys.argv[1].strip())
    array_size = int(sys.argv[2].strip())
    input_file = sys.argv[3].strip()
    output_file = sys.argv[4].strip()
    start = time()
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    if process == 0:
        # Then we launder
        print("-" * 100)
        print(f"Laundering began at {date}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        laundered_data = launder(
            array_size, input_hist, input_bin_edges, input_bin_centers
        )
        laundered_t = convert_z_to_t(laundered_data)
        np.savetxt(output_file, laundered_t)
        print(f"Laundering completed in {time() - start:.3f} seconds")
        print(f"Data in {input_file} laundered and saved to {output_file}")
        print("-" * 100)
    else:
        # Then we symmetrise a histogram
        print("-" * 100)
        print(f"Symmetrisation began at {date}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        symmetrized_data = center_z_distribution(input_hist, input_bin_edges)
        save_data(symmetrized_data, input_bin_edges, input_bin_centers, output_file)
        print(f"Symmetrisation completed in {time() - start:.3f} seconds")
        print(f"Data in {input_file} symmetrised and saved to {output_file}")
        print("-" * 100)
    # Delete old files once done to prevent buildup
    # os.remove(input_file)
