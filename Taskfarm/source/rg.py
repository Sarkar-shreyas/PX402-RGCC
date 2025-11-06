#!/usr/bin/env python
"""
This file will load the histogram from a previous rg step and the current step
Then, it'll compute the L2 distance, mean and std, returning those values, along with whether its converged or not.
"""

import numpy as np
import sys
from .utilities import (
    l2_distance,
    DIST_TOLERANCE,
    STD_TOLERANCE,
    hist_moments,
)


if __name__ == "__main__":
    rg_step = int(sys.argv[1].strip())
    old_hist_file = sys.argv[2].strip()
    current_hist_file = sys.argv[3].strip()
    output_file = sys.argv[4].strip()
    current_hist_data = np.load(current_hist_file)
    current_hist_vals = current_hist_data["histval"]
    current_hist_bins = current_hist_data["binedges"]
    current_mean, current_std = hist_moments(current_hist_vals, current_hist_bins)
    converged = False

    if rg_step == 0 or old_hist_file.lower() == "none":
        np.savez_compressed(
            output_file,
            l2_dist=np.nan,
            mean=current_mean,
            std=current_std,
            converged=False,
        )
        print(
            f"Moments of current histogram for RG step {rg_step} saved to {output_file}"
        )
        sys.exit(0)
    else:
        old_hist_data = np.load(old_hist_file)
        old_hist_vals = old_hist_data["histval"]
        old_hist_bins = old_hist_data["binedges"]
        old_mean, old_std = hist_moments(old_hist_vals, old_hist_bins)
        distance = l2_distance(
            old_hist_vals, current_hist_vals, old_hist_bins, current_hist_bins
        )

        std_diff = current_std - old_std
        if distance <= DIST_TOLERANCE and np.abs(std_diff) <= STD_TOLERANCE:
            converged = True

        np.savez_compressed(
            output_file,
            l2_dist=distance,
            mean=current_mean,
            std=current_std,
            converged=converged,
        )
        print(
            f"Moments of current histogram for RG step {rg_step} saved to {output_file}"
        )
