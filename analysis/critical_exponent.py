"""
Critical exponent (ν) extraction and peak analysis for RG simulation outputs.

This script post-processes simulation data to estimate the critical exponent ν.
It supports both new-format (config-based) and legacy (no config) data, detecting
and adapting to the available metadata. Outputs include ν estimates and diagnostic
plots, written to the appropriate stats/ and plots/ subfolders.

- New-format: Reads config YAML for run parameters.
- Old-format: Falls back to heuristics and folder/filename parsing.

Assumption: Data layout and file naming follow conventions described in the repo docs.
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from source.utilities import calculate_nu, get_density, hist_moments, launder, build_rng
from source.config import load_yaml, build_config
from scipy.stats import norm
from source.fitters import estimate_z_peak, fit_z_peaks
from analysis.data_plotting import (
    load_hist_data,
    construct_moments_dict,
    plot_data,
    calculate_average_nu,
    build_plot_parser,
    build_config_path,
)
import os
import json
from time import time
from constants import data_dir, SHIFTS, config_file, local_dir

TYPE = "EXP"


def slice_middle(
    counts: np.ndarray,
    bins: np.ndarray,
    centers: np.ndarray,
    densities: np.ndarray,
    shift: float,
) -> tuple:
    """
    Extract the central region of a Gaussian-like histogram, bounded by [-25+shift, 25+shift].

    Args:
        counts (np.ndarray): Histogram bin counts.
        bins (np.ndarray): Histogram bin edges.
        centers (np.ndarray): Bin centers.
        densities (np.ndarray): Histogram densities.
        shift (float): Value to shift the window.

    Returns:
        tuple: (counts, bins, centers, densities) arrays for the sliced region.
    """
    mask = np.logical_and((centers >= -25.0 + shift), (centers <= 25.0 + shift))
    indexes = np.where(mask)[0]
    starting_index = indexes[0]
    ending_index = indexes[-1]
    return (
        counts[starting_index : ending_index + 1],
        bins[starting_index : ending_index + 2],
        centers[starting_index : ending_index + 1],
        densities[starting_index : ending_index + 1],
    )


def main():
    """
    Main entry point for critical exponent (ν) extraction and plotting.

    This script detects whether a config file is present (new-format run) or not (legacy run).
    - If config is present, it parses run parameters from the config YAML.
    - If config is absent, it falls back to legacy heuristics (parsing version/steps from folder names or filenames).
    - Loads RG run parameters and data, performs peak estimation, and produces ν estimates and diagnostic plots.

    Args:
        None. Uses CLI arguments (see build_plot_parser for options).

    Returns:
        None. Side effects: writes plots and stats to output folders.

    Raises:
        SystemExit: On usage error or missing data.

    Notes:
        - If config is missing, uses folder/filename parsing for legacy runs.
        - Output files are written to stats/ and plots/ subfolders.
        - Assumption: old-format data uses legacy naming conventions.
    """
    parser = build_plot_parser()
    args = parser.parse_args()
    if os.path.exists(args.loc):
        config_path = build_config_path(args.loc, args.version, args.mode)
    else:
        config_path = str(config_file)
    config = load_yaml(config_path)
    rg_config = build_config(config)
    seed = rg_config.seed
    rng = build_rng(seed)
    sampler = rg_config.resample
    version = str(args.version)
    num_rg = int(args.steps)
    rg = num_rg + 1
    if str(args.loc).strip().lower() == "local":
        data_folder = local_dir
    else:
        data_folder = data_dir
    main_dir = f"{data_folder}/{version}"
    stats_dir = f"{data_folder}/{version}/{TYPE}/stats"
    plots_dir = f"{data_folder}/{version}/{TYPE}/plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    data_map = defaultdict(dict)
    vars = ["t", "r", "f", "tau", "leak", "surv", "z"]
    print(f"Performing peak estimation for {version}")
    print("=" * 100)
    # Load the FP distribution
    fp_file = f"{data_folder}/{version}/FP/hist/z/sym_z_hist_RG{num_rg - 1}.npz"
    fp_counts, fp_bins, fp_centers = load_hist_data(fp_file)
    fp_density = get_density(fp_counts, fp_bins)

    # Get all the initial plots made for inspection
    start = time()
    for shift in SHIFTS:
        for var in vars:
            data_map[shift][var] = []
            shift_dir = f"{data_folder}/{version}/{TYPE}/hist/{shift}/{var}"
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
                    filename = f"{shift_dir}/{var}_hist_RG{i - 1}.npz"
                else:
                    filename = f"{shift_dir}/{var}_hist_RG{i - 1}.npz"
                counts, bins, centers = load_hist_data(filename)
                densities = get_density(counts, bins)
                data_map[shift][var].append([counts, bins, centers, densities])
            filename = f"{shift_plot_dir}/{var}_hist_{shift}.png"
            plot_data(var, filename, data_map[shift][var], TYPE, num_rg)
        construct_moments_dict(
            shift_stats_dir, shift_plot_dir, vars, data_map[shift], num_rg
        )

        print(f"Plots for shift {shift} have been made.")
        print(f"Stats for shift {shift} have been made.")
        print("-" * 100)
    print("=" * 100)
    # print(data_map.keys())
    fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_0.set_xlim([0, float(max(SHIFTS)) + SHIFTS[1]])
    # ax_0.set_ylim([0.0, 2])
    ax_0.set_title("Scatter plot of z peaks")
    ax_0.set_xlabel("z_0")
    ax_0.set_ylabel("z_peak")
    ax_1.set_title("Scatter plot and line fit of z peaks")
    ax_1.set_xlabel("z_0")
    ax_1.set_ylabel("z_peak")
    ax_1.set_xlim([0, float(max(SHIFTS)) + SHIFTS[1]])
    # ax_1.set_ylim([0, 2])

    peaks = np.zeros((rg, len(SHIFTS))).astype(float)
    min_peaks = np.zeros((rg, len(SHIFTS))).astype(float)
    max_peaks = np.zeros((rg, len(SHIFTS))).astype(float)
    peak_errs = np.zeros((rg, len(SHIFTS))).astype(float)
    means = np.zeros((rg, len(SHIFTS))).astype(float)
    stds = np.zeros((rg, len(SHIFTS))).astype(float)
    print("Beginning peak estimations")
    print("-" * 100)
    for j in range(len(SHIFTS)):
        shift = SHIFTS[j]
        shift_val = float(shift)
        print(f"Estimating peak for shift {shift}")
        # peaks[0, j] = estimate_z_peak(fp_counts, fp_bins, fp_centers)
        # means[0, j], stds[0, j] = hist_moments(fp_counts, fp_bins)
        peaks[0, j] = 0.0
        peak_errs[0, j] = 0.0
        means[0, j] = 0.0
        for i in range(1, rg):
            counts = data_map[shift]["z"][i][0]
            bins = data_map[shift]["z"][i][1]
            centers = data_map[shift]["z"][i][2]
            densities = data_map[shift]["z"][i][3]
            sliced_counts, sliced_bins, sliced_centers, sliced_densities = slice_middle(
                counts, bins, centers, densities, shift_val
            )
            mean, std = hist_moments(sliced_counts, sliced_bins)
            # test = launder(1000000, sliced_counts, sliced_bins, sliced_centers, rng)

            # test_mu, test_std = norm.fit(test)
            # print(f"Fitted mu = {test_mu}, Calculated mean = {mean}")
            min_peaks[i, j], max_peaks[i, j], peaks[i, j] = estimate_z_peak(
                sliced_counts, sliced_bins, sliced_centers, rng, sampler
            )
            peak_errs[i, j] = max_peaks[i, j] - min_peaks[i, j]
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
    peak_data = defaultdict(dict)
    peak_data_file = f"{main_dir}/peaks.json"
    overall_stats_file = f"{main_dir}/overall_stats.json"
    x = np.array(SHIFTS).astype(float)
    nus = []
    other_nus = []
    r2s = []
    other_r2s = []
    min_nus = []
    max_nus = []
    nu_errors = []
    starting_index = 1
    # rgs = [i + 1 for i in range(rg)]
    for i in range(starting_index, rg):
        # Without subtracting anything
        # y = peaks[i, :]
        # m = means[i, :]
        # min_y = min_peaks[i, :]
        # max_y = max_peaks[i, :]

        # Subtracting the peaks for the Fixed point distribution to re-center bins
        # y = peaks[i, :] - peaks[0, :]
        # m = means[i, :] - means[0, :]
        # min_y = min_peaks[i, :] - min_peaks[0, :]
        # max_y = max_peaks[i, :] - max_peaks[0, :]

        # Subtracting the peaks for shift=0.0
        y = peaks[i, :] - peaks[i, 0]
        m = means[i, :] - means[i, 0]
        min_y = min_peaks[i, :] - min_peaks[i, 0]
        max_y = max_peaks[i, :] - max_peaks[i, 0]
        # y -= peaks[i, 0]
        # m -= means[i, 0]
        # min_y -= min_peaks[i, 0]
        # max_y -= max_peaks[i, 0]
        x_fit = x[:]
        y_fit = y[:]
        m_fit = m[:]
        # print(f"For RG{i}: Mean diffs: {m}")
        ms, mr2 = fit_z_peaks(x_fit, m_fit)
        slope, r2 = fit_z_peaks(x_fit, y_fit)
        ax_0.set_title("Means")
        ax_1.set_title("Estimated peaks")
        if i in (1, 2, 3, 4, 5, 6, 7):
            ax_0.scatter(x_fit[1:], m_fit[1:])
            ax_0.plot(x, ms * x, label=f"RG_{i}")
            # ax_1.scatter(x_fit, y_fit)
            e = ax_1.errorbar(
                x_fit[1:],
                y_fit[1:],
                yerr=peak_errs[i, 1:],
                marker="o",
                linestyle="none",
                capsize=2.5,
            )
            c = e[0].get_color()
            # ax_1.set_ylim((0.0, 0.01))
            x_line = np.linspace(0, float(max(SHIFTS)) + SHIFTS[1], 200)
            y_line = slope * x_line
            ax_1.plot(x_line, y_line, label=f"RG_{i}", color=c)

        nu = calculate_nu(slope, i)
        other_nu = calculate_nu(ms, i)

        min_slope, min_r2 = fit_z_peaks(x_fit, min_y[:])
        max_slope, max_r2 = fit_z_peaks(x_fit, max_y[:])
        min_nus.append(calculate_nu(min_slope, i))
        max_nus.append(calculate_nu(max_slope, i))
        nu_errors.append(np.abs(max_nus[i - 1] - min_nus[i - 1]))
        nus.append(nu)
        other_nus.append(other_nu)
        r2s.append(r2)
        other_r2s.append(mr2)
        peak_data[f"RG{i}"] = {
            "Peaks": list(peaks[i, :]),
            "Min Peaks": list(min_peaks[i, :]),
            "Max Peaks": list(max_peaks[i, :]),
            "Peak Errors": list(peak_errs[i, :]),
        }
        overall_stats[f"RG{i}"] = {
            "Peak Nu": float(nu),
            "Mean Nu": float(other_nu),
            "Peak Slope": float(slope),
            "Mean Slope": float(ms),
            "Peak R2": float(r2),
            "Mean R2": float(mr2),
        }

    print("=" * 100)
    ax_0.legend()
    ax_1.legend()
    z_peaks_plot = f"{main_dir}/z_peaks.png"
    Nu_plot = f"{main_dir}/Nu_{len(SHIFTS)}_shifts.png"
    plt.savefig(z_peaks_plot, dpi=150)
    plt.close()
    with open(overall_stats_file, "w") as f:
        json.dump(overall_stats, f, indent=2)
    with open(peak_data_file, "w") as f:
        json.dump(peak_data, f, indent=2)
    print(f"Overall stats for z saved to {overall_stats_file}")
    print(f"Peak data saved to {peak_data_file}")
    print(f"z peaks data plotted and saved to {z_peaks_plot}")
    system_size = [2**i for i in range(starting_index, rg)]
    fig, (ax_2, ax_3) = plt.subplots(1, 2, figsize=(10, 4))
    # ax_2.set_xlim([0, 0.01])
    ax_2.set_ylim([2.3, 2.8])
    ax_2.set_title("Scatter plot of Nu vs System size from means")
    ax_2.set_xlabel("2^n")
    ax_2.set_ylabel("Nu")
    # ax_2.set_xticks(system_size, system_size)
    ax_3.set_title("Scatter plot of Nu vs System size from peaks")
    ax_3.set_xlabel("2^n")
    ax_3.set_ylabel("Nu")
    # ax_3.set_xticks(system_size, system_size)
    # ax_3.set_xlim([0, 0.01])
    ax_3.set_ylim([2, 5])
    # ind = 2
    ax_2.scatter(system_size, other_nus)
    ax_3.errorbar(
        system_size[:],
        nus[:],
        yerr=nu_errors[:],
        marker="o",
        linestyle="none",
        capsize=3.0,
        markersize=4.0,
    )

    # ax_3.scatter(system_size, nus)
    plt.savefig(Nu_plot, dpi=150)
    plt.close()
    print(f"Nu data plotted and saved to {Nu_plot}")
    print("-" * 100)
    calculate_average_nu(overall_stats, 7, rg)
    print("-" * 100)
    print(f"Analysis done after {time() - start:.3f} seconds")


if __name__ == "__main__":
    main()
