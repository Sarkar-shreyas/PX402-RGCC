#!/usr/bin/env python
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from source.utilities import (
    calculate_nu,
    get_density,
    hist_moments,
)
from source.fitters import estimate_z_peak, fit_z_peaks
from data_plotting import load_hist_data, construct_moments_dict, plot_data
import os
import json
from time import time

DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm"
CURRENT_VERSION = "1.82C"
TYPE = "EXP"
NUM_RG = 9
# SHIFTS = [0.0, 0.003, 0.007, 0.009]
SHIFTS = [0.0, 0.003, 0.005, 0.007, 0.009]
if __name__ == "__main__":
    version = CURRENT_VERSION
    rg = NUM_RG + 1
    stats_dir = f"{DATA_DIR}/v{version}/{TYPE}/stats"
    plots_dir = f"{DATA_DIR}/v{version}/{TYPE}/plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    data_map = defaultdict(dict)
    vars = ["t", "input_t", "z"]

    # Load the FP distribution
    fp_file = f"{DATA_DIR}/v{version}/FP/hist/sym_z/sym_z_hist_RG{NUM_RG - 1}.npz"
    fp_counts, fp_bins, fp_centers = load_hist_data(fp_file)
    fp_density = get_density(fp_counts, fp_bins)

    # Get all the initial plots made for inspection
    start = time()
    for shift in SHIFTS:
        # data_map[shift]["fp"] = [fp_counts, fp_bins, fp_centers, fp_density]
        for var in vars:
            data_map[shift][var] = []
            shift_dir = f"{DATA_DIR}/v{version}/{TYPE}/shift_{shift}/hist/{var}"
            shift_plot_dir = f"{plots_dir}/{shift}"
            shift_stats_dir = f"{stats_dir}/{shift}"
            os.makedirs(shift_plot_dir, exist_ok=True)
            os.makedirs(shift_stats_dir, exist_ok=True)
            if var == "z":
                data_map[shift]["z"].append(
                    [fp_counts, fp_bins, fp_centers, fp_density]
                )
            for i in range(1, rg):
                if var == "z":
                    filename = f"{shift_dir}/{var}_hist_unsym_RG{i - 1}.npz"
                else:
                    filename = f"{shift_dir}/{var}_hist_RG{i - 1}.npz"
                counts, bins, centers = load_hist_data(filename)
                # if var == "z":
                #     print(
                #         f"For RG{i}, first bin = {bins[0]}, first center = {centers[0]}, last bin = {bins[-1]}, last center = {bins[-1]}"
                #     )
                #     print(f"Min: {min(counts)}, Max: {max(counts)}")
                densities = get_density(counts, bins)
                data_map[shift][var].append([counts, bins, centers, densities])
            # print(f"All histograms for shift {shift} have been plotted")
            filename = f"{shift_plot_dir}/{var}_hist_shift_{shift}.png"
            plot_data(var, filename, data_map[shift][var], TYPE)
        # plot_data(
        #     "z",
        #     f"{shift_plot_dir}/fp_hist_shift_{shift}.png",
        #     data_map[shift]["fp"],
        #     TYPE,
        # )
        # print("Data for fp plotted")
        # construct_moments_dict(shift_stats_dir, shift_plot_dir, ["fp"], data_map[shift])
        # print("Stats for fp saved")
        construct_moments_dict(shift_stats_dir, shift_plot_dir, vars, data_map[shift])

        print(f"Plots for shift {shift} have been made.")
        print(f"Stats for shift {shift} have been made.")
        print("-" * 100)
    print("=" * 100)
    # print(data_map.keys())
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

    peaks = np.zeros((rg, len(SHIFTS))).astype(float)
    means = np.zeros((rg, len(SHIFTS))).astype(float)
    stds = np.zeros((rg, len(SHIFTS))).astype(float)
    print("Beginning peak estimations")
    print("-" * 100)

    for j in range(len(SHIFTS)):
        z_moments = []
        z_dist = []
        shift = SHIFTS[j]
        print(f"Estimating peak for shift {shift}")
        loop = time()
        # peaks[0, j] = estimate_z_peak(fp_counts, fp_bins, fp_centers)
        # means[0, j], stds[0, j] = hist_moments(fp_counts, fp_bins)
        peaks[0, j] = 0.0
        means[0, j] = 0.0
        for i in range(1, rg):
            counts = data_map[shift]["z"][i][0]
            bins = data_map[shift]["z"][i][1]
            centers = data_map[shift]["z"][i][2]
            densities = data_map[shift]["z"][i][3]
            mean, std = hist_moments(counts, bins)
            peaks[i, j] = estimate_z_peak(counts, bins, centers)
            means[i, j] = mean
            stds[i, j] = std
        # print(
        #     f"Shift {shift}, RG {i}: Min bin = {min(bins)}, Max bin = {max(bins)}, Min center = {min(centers)}, Max center = {max(centers)}"
        # )
        # print("-" * 100)
        print(f"Peak estimated for shift {shift} after {time() - start:.3f} seconds")
    print("Finished peaks estimation for every shift")
    print("=" * 100)
    # print(z_moments)
    overall_stats = defaultdict(dict)
    overall_stats_file = f"{stats_dir}/overall_stats.json"
    x = np.array(SHIFTS).astype(float)
    nus = []
    other_nus = []
    r2s = []
    other_r2s = []

    # rgs = [i + 1 for i in range(rg)]
    for i in range(1, rg):
        y = peaks[i, :] - peaks[0, :]
        m = means[i, :] - means[0, :]

        x_fit = x[1:]
        y_fit = y[1:]
        m_fit = m[1:]
        # print(f"For RG{i}: Mean diffs: {m}")
        ms, mr2 = fit_z_peaks(x_fit, m_fit)
        slope, r2 = fit_z_peaks(x_fit, y_fit)
        ax_0.set_title("Means")
        ax_1.set_title("Estimated peaks")
        # if i % 2 == 0:
        ax_0.scatter(x_fit, m_fit)
        ax_0.plot(x, ms * x, label=f"RG_{i}")
        ax_1.scatter(x_fit, y_fit)
        ax_1.plot(x, slope * x, label=f"RG_{i}")
        nu = calculate_nu(slope, i)
        other_nu = calculate_nu(ms, i)
        nus.append(nu)
        other_nus.append(other_nu)
        r2s.append(r2)
        other_r2s.append(mr2)
        overall_stats[f"RG{i}"] = {
            "Peak Nu": float(nu),
            "Mean Nu": float(other_nu),
            "Peak Slope": float(slope),
            "Mean Slope": float(ms),
            "Peak R2": float(r2),
            "Mean R2": float(mr2),
        }

    ax_0.legend()
    ax_1.legend()
    z_peaks_plot = f"{plots_dir}/z_peaks.png"
    Nu_plot = f"{plots_dir}/Nu_{len(SHIFTS)}_shifts.png"
    plt.savefig(z_peaks_plot, dpi=150)
    plt.close()
    with open(overall_stats_file, "w") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"Overall stats for z saved to {overall_stats_file}")
    print(f"z peaks data plotted and saved to {z_peaks_plot}")
    system_size = [2**i for i in range(rg - 1)]
    fig, (ax_2, ax_3) = plt.subplots(1, 2, figsize=(10, 4))
    # ax_2.set_xlim([0, 0.01])
    # ax_2.set_ylim([0.0, 2])
    ax_2.set_title("Scatter plot of Nu vs System size from means")
    ax_2.set_xlabel("2^n")
    ax_2.set_ylabel("Nu")
    ax_2.set_xticks(system_size, system_size)
    ax_3.set_title("Scatter plot of Nu vs System size from peaks")
    ax_3.set_xlabel("2^n")
    ax_3.set_ylabel("Nu")
    ax_3.set_xticks(system_size, system_size)
    # ax_3.set_xlim([0, 0.01])
    # ax_3.set_ylim([0, 2])
    # ind = 2

    ax_2.scatter(system_size, other_nus[:])
    ax_3.scatter(system_size, nus[:])
    plt.savefig(Nu_plot, dpi=150)
    plt.close()
    print(f"Nu data plotted and saved to {Nu_plot}")
    print(f"Analysis done after {time() - start:.3f} seconds")
