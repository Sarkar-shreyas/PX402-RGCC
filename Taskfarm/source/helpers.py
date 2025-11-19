#!/usr/bin/env python
"""
This is a placeholder file meant to let me directly either apply laundering to an input array,
or symmetrise an input histogram for convenience
"""

import numpy as np
import sys
from time import time
from datetime import datetime, timezone
from .utilities import (
    save_data,
    launder,
    center_z_distribution,
    convert_z_to_t,
    convert_t_to_z,
)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(
            " Usage: helpers.py PROCESS ARRAY_SIZE INPUT_FILE OUTPUT_FILE \n"
            " PROCESS 0 : Launder from z-histogram + Convert to t \n"
            " PROCESS 1 : Symmetrise z-histogram \n"
            " PROCESS 2 : Launder from t-histogram \n"
            " PROCESS 3 : Convert t to z \n"
        )
    process = int(sys.argv[1].strip())
    array_size = int(sys.argv[2].strip())
    input_file = sys.argv[3].strip()
    output_file = sys.argv[4].strip()
    start = time()
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if process == 0:
        # Launder from input z histogram and convert to t data
        print("-" * 100)
        print(f"[{date}]: Began Laundering from z-histogram at {input_file}")
        input_data = np.load(input_file)
        input_hist = input_data["histval"]
        input_bin_edges = input_data["binedges"]
        input_bin_centers = input_data["bincenters"]
        laundered_data = launder(
            array_size, input_hist, input_bin_edges, input_bin_centers
        )
        laundered_t = convert_z_to_t(laundered_data)
        laundered_t = laundered_t.astype(np.float32)
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
            array_size, input_hist, input_bin_edges, input_bin_centers
        )
        laundered_t = laundered_t.astype(np.float32)
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
        z = z.astype(np.float32)
        np.save(output_file, z)
        print(f"Conversion completed in {time() - start:.3f} seconds")
        print(f"t data in {input_file} converted and saved as z data to {output_file}")
        print("-" * 100)
    else:
        raise SystemExit("Invalid PROCESS entered. Valid: [0, 1, 2, 3]")
    # Delete old files once done to prevent buildup
    # os.remove(input_file)
