#!/usr/bin/env python
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from source.utilities import (
    estimate_z_peak,
    fit_z_peaks,
    calculate_nu,
    get_density,
    hist_moments,
    l2_distance,
)
from data_plotting import load_hist_data, construct_moments_dict
import os
import json
from time import time

DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm"
CURRENT_VERSION = 1.4
TYPE = "EXP"
NUM_RG = 7
SHIFTS = [0.0, 0.003, 0.005, 0.007, 0.009]

if __name__ == "__main__":
    version = CURRENT_VERSION
    stats_dir = f"{DATA_DIR}/v{version}/{TYPE}/stats"
    plots_dir = f"{DATA_DIR}/v{version}/{TYPE}/plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    map = defaultdict(dict)
    vars = ["t", "g", "input_t", "z"]
    # Get all the initial plots made for inspection
    start = time()
    for var in vars:
        for shift in SHIFTS:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
            map[var][shift] = []
            ax0.set_title(f"Unclipped histogram of {var} data for shift {shift}")
            ax0.set_xlabel(f"{var}")
            ax0.set_ylabel(f"P{var}")
            ax1.set_title(f"Clipped histogram of {var} data for shift {shift}")
            ax1.set_xlabel(f"{var}")
            ax1.set_ylabel(f"P({var})")
            if var == "z":
                ax0.set_xlim([-50, 50])
                ax0.set_ylim([0.0, 0.3])
                ax1.set_xlim([-25, 25.0])
                ax1.set_ylim([0, 0.3])
            else:
                ax0.set_ylim([0.0, 2.0])
                ax1.set_ylim([0.0, 2.0])
            shift_dir = f"{DATA_DIR}/v{version}/{TYPE}/shift_{shift}/hist/{var}"
            shift_plot_dir = f"{plots_dir}/{shift}"
            # shift_stats_dir = f"{stats_dir}/{shift}"
            os.makedirs(shift_plot_dir, exist_ok=True)
            # os.makedirs(shift_stats_dir, exist_ok=True)
            for i in range(NUM_RG):
                filename = f"{shift_dir}/{var}_hist_RG{i}.npz"
                counts, bins, centers = load_hist_data(filename)
                densities = get_density(counts, bins)
                map[var][shift].append([counts, bins, centers, densities])
                ax0.plot(centers, densities, label=f"RG_{i}")
                ax1.plot(centers, densities, label=f"RG_{i}")
            ax0.legend()
            ax1.legend()
            plt.savefig(f"{shift_plot_dir}/{var}_hist_shift_{shift}.png", dpi=150)
            plt.close()
            # print(f"All histograms for shift {shift} have been plotted")
        print(f"Finished plotting data of {var} for all shifts")

    print("All plots made.")
    # print(map.keys())
    fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_0.set_xlim([0, 0.01])
    # ax_0.set_ylim([0.0, 2])
    ax_0.set_title("Scatter plot of z peaks")
    ax_0.set_xlabel("z_0")
    ax_0.set_ylabel("z_peak")
    ax_1.set_title("Scatter plot and line fit of z peaks")
    ax_1.set_xlabel("z_0")
    ax_1.set_ylabel("z_peak")
    ax_1.set_xlim([0, 0.01])
    # ax_1.set_ylim([0, 2])

    peaks = np.zeros((NUM_RG, len(SHIFTS))).astype(float)
    print("Beginning peak estimations")
    print("-" * 100)
    means = np.zeros((NUM_RG, len(SHIFTS))).astype(float)
    for j in range(len(SHIFTS)):
        z_moments = []
        z_dist = []
        shift = SHIFTS[j]
        # print(f"Estimating peak for shift {shift}")
        loop = time()
        for i in range(NUM_RG):
            counts = map["z"][shift][i][0]
            bins = map["z"][shift][i][1]
            centers = map["z"][shift][i][2]
            densities = map["z"][shift][i][3]
            peaks[i, j] = estimate_z_peak(densities, bins, centers)
            mean, std = hist_moments(counts, bins)
            means[i, j] = mean
            z_moments.append([mean, std])
            if i > 0:
                old_counts = map["z"][shift][i - 1][0]
                old_bins = map["z"][shift][i - 1][1]
                old_std = z_moments[i - 1][1]
                l2 = l2_distance(old_counts, counts, old_bins, bins)
                z_dist.append(l2)
            # print(
            #     f"Peak estimated for RG{i} of shift {shift} after {time() - loop:.3f} seconds"
            # )
        z_stats = f"{stats_dir}/{shift}_z_stats.json"
        construct_moments_dict(z_stats, z_moments, z_dist, NUM_RG)
        print(f"Stats for shift {shift} saved to {z_stats}")
        print("-" * 100)
    print(
        f"Peaks estimated for each shift at every RG {time() - start:.3f} seconds from start of program"
    )
    # print(z_moments)
    overall_stats = defaultdict(dict)
    overall_stats_file = f"{stats_dir}/overall_stats.json"
    x = np.array(SHIFTS).astype(float)
    nus = []
    other_nus = []
    r2s = []
    other_r2s = []
    for i in range(NUM_RG):
        y = np.abs(peaks[i, :] - peaks[i, 0])
        m = means[i, :] - means[i, 0]
        # print(f"For RG{i}: Mean diffs: {m}")
        ms, mr2 = fit_z_peaks(x, m)
        slope, r2 = fit_z_peaks(x, y)
        ax_0.set_title("Difference of means")
        ax_1.set_title("Difference of estimated peaks")
        if i % 2 == 0:
            ax_0.scatter(x, m)
            ax_0.plot(x, ms * x, label=f"RG_{i}")
            ax_1.scatter(x, y)
            ax_1.plot(x, slope * x, label=f"RG_{i}")
        nu = calculate_nu(slope, i)
        other_nu = calculate_nu(ms, i)
        nus.append(nu)
        other_nus.append(other_nu)
        r2s.append(r2)
        other_r2s.append(mr2)
        overall_stats[f"RG{i}"] = {
            "Nu": float(nu),
            "Other Nu": float(other_nu),
            "Slope": float(slope),
            "Mean Slope": float(ms),
            "R2": float(r2),
            "Mean R2": float(mr2),
            "Peaks": list(peaks[i, :]),
            "Peak diffs": list(y),
            "Means": list(means[i, :]),
            "Mean diffs": list(m),
        }
    ax_0.legend()
    ax_1.legend()
    plt.savefig(f"{plots_dir}/z_peaks.png", dpi=150)
    plt.close()
    with open(overall_stats_file, "w") as f:
        json.dump(overall_stats, f, indent=2)

    fig, (ax_2, ax_3) = plt.subplots(1, 2, figsize=(10, 4))
    # ax_2.set_xlim([0, 0.01])
    # ax_2.set_ylim([0.0, 2])
    ax_2.set_title("Scatter plot of Nu vs System size from means")
    ax_2.set_xlabel("2^n")
    ax_2.set_ylabel("Nu")
    ax_3.set_title("Scatter plot of Nu vs System size from peaks")
    ax_3.set_xlabel("2^n")
    ax_3.set_ylabel("Nu")
    # ax_3.set_xlim([0, 0.01])
    # ax_3.set_ylim([0, 2])
    ind = 2
    system_size = [2**i for i in range(ind, NUM_RG)]
    ax_2.scatter(system_size, other_nus[ind:])
    ax_3.scatter(system_size, nus[ind:])
    plt.savefig(f"{plots_dir}/Nu.png", dpi=150)
    plt.close()
    print(f"Analysis done after {time() - start:.3f} seconds")
