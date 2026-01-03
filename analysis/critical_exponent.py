from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from source.utilities import calculate_nu, get_density, hist_moments, launder, build_rng
from source.config import load_yaml, build_config
from scipy.stats import norm
from source.fitters import estimate_z_peak, fit_z_peaks
from data_plotting import (
    load_hist_data,
    construct_moments_dict,
    plot_data,
    calculate_average_nu,
)
import os
import json
from time import time
from constants import data_dir, CURRENT_VERSION, NUM_RG, SHIFTS

TYPE = "EXP"


def slice_middle(
    counts: np.ndarray,
    bins: np.ndarray,
    centers: np.ndarray,
    densities: np.ndarray,
    shift: float,
) -> tuple:
    """Slice out the middle of a gaussian histogram"""
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
    config = load_yaml(f"{data_dir}")
    rg_config = build_config(config)
    seed = rg_config.seed
    rng = build_rng(seed)
    sampler = rg_config.resample
    version = CURRENT_VERSION
    rg = NUM_RG + 1
    main_dir = f"{data_dir}/{version}"
    stats_dir = f"{data_dir}/{version}/{TYPE}/stats"
    plots_dir = f"{data_dir}/{version}/{TYPE}/plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    data_map = defaultdict(dict)
    vars = ["t", "input_t", "z"]
    print(f"Performing peak estimation for {CURRENT_VERSION}")
    print("=" * 100)
    # Load the FP distribution
    fp_file = f"{data_dir}/{version}/FP/hist/sym_z/sym_z_hist_RG{NUM_RG - 1}.npz"
    fp_counts, fp_bins, fp_centers = load_hist_data(fp_file)
    fp_density = get_density(fp_counts, fp_bins)

    # Get all the initial plots made for inspection
    start = time()
    for shift in SHIFTS:
        # data_map[shift]["fp"] = [fp_counts, fp_bins, fp_centers, fp_density]
        for var in vars:
            data_map[shift][var] = []
            shift_dir = f"{data_dir}/{version}/{TYPE}/shift_{shift}/hist/{var}"
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
    ax_1.set_xlim([0, float(max(SHIFTS)) + 0.001])
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
            test = launder(1000000, sliced_counts, sliced_bins, sliced_centers, rng)
            test_mu, test_std = norm.fit(test)
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
            x_line = np.linspace(0, 0.01, 200)
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
