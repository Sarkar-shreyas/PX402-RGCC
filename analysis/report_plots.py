"""
Report-quality plotting for RG simulation analysis outputs.

This script generates publication-ready plots from processed RG simulation data.
It supports both new-format (config-based) and legacy (no config) data, detecting
and adapting to the available metadata. Plots are saved to the appropriate output folders.

- New-format: Reads config YAML for run parameters.
- Old-format: Falls back to heuristics and folder/filename parsing.

Assumption: Data layout and file naming follow conventions described in the repo docs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
from source.fitters import fit_z_peaks
from source.utilities import calculate_nu, get_density, hist_moments
from source.config import build_config, load_yaml
from analysis.data_plotting import build_plot_parser, build_config_path
from constants import data_dir, SHIFTS, config_file
import os
import json
from scipy.stats import norm

markers = ["*", "+", "d", "v", "s", "p"]


def plot_z_fp():
    # Load FP z data from v1.90S
    z_fp = np.load(f"{data_dir}/v1.90S/FP/hist/sym_z/sym_z_hist_RG8.npz")
    z_bins = z_fp["bincenters"]
    z_vals = z_fp["histval"]
    z_densities = get_density(z_vals, z_fp["binedges"])

    # Load FP moments from v1.90S
    z_moments = json.load(open(f"{data_dir}/v1.90S/FP/stats/sym_z_moments.json", "r"))

    mean = z_moments["RG_8"]["mean"]
    std = z_moments["RG_8"]["std"]
    plt.figure("fp")
    plt.plot(
        z_bins,
        norm.pdf(z_bins, loc=mean, scale=std),
        linestyle="--",
        color="b",
        alpha=0.6,
        label="Gaussian",
    )
    plt.plot(z_bins, z_densities, color="g", alpha=0.8)
    plt.scatter(
        z_bins[::200],
        z_densities[::200],
        color="g",
        label="n = 9",
        marker="o",
        facecolor="none",
    )
    plt.xlim((-5.0, 5.0))
    plt.xlabel(r"$z$")
    plt.ylabel(r"$P(z)$")
    # plt.ylim((0, 0.22))
    plt.legend()
    plt.savefig("./report/z_FP.pdf")
    plt.close("fp")


def plot_z_peaks():
    # Load z peaks from v1.90S
    peaks_data = json.load(open(f"{data_dir}/v1.90S/peaks.json", "r"))
    num_rg = 12
    peaks = {}
    errors = {}
    for i in range(1, num_rg + 1):
        rg_peaks = peaks_data[f"RG{i}"]["Peaks"]
        rg_peaks_errs = peaks_data[f"RG{i}"]["Peak Errors"]
        for j in range(len(SHIFTS)):
            if SHIFTS[j] not in peaks:
                peaks.update({SHIFTS[j]: [rg_peaks[j]]})
            else:
                peaks[SHIFTS[j]].append(rg_peaks[j])
            if SHIFTS[j] not in errors:
                errors.update({SHIFTS[j]: [rg_peaks_errs[j]]})
            else:
                errors[SHIFTS[j]].append(rg_peaks_errs[j])
    # print(peaks)
    shifts = np.array([float(shift) for shift in SHIFTS])
    nus = []
    min_nus = []
    max_nus = []
    nu_errs = []
    plt.figure("peaks")
    for i in range(1, 10):
        y = []
        err = []
        for shift in shifts[1:]:
            y.append(peaks[f"{shift}"][i - 1] - peaks["0.0"][i - 1])
            err.append(errors[f"{shift}"][i - 1])
        e = plt.errorbar(
            shifts[1:],
            y,
            yerr=err,
            marker="o",
            linestyle="none",
            capsize=2.5,
        )
        c = e[0].get_color()
        slope, r2 = fit_z_peaks(np.array(shifts), peaks_data[f"RG{i}"]["Peaks"])
        min_slope, min_r2 = fit_z_peaks(
            np.array(shifts), peaks_data[f"RG{i}"]["Min Peaks"]
        )
        max_slope, max_r2 = fit_z_peaks(
            np.array(shifts), peaks_data[f"RG{i}"]["Max Peaks"]
        )
        nus.append(calculate_nu(slope, i))
        min_nus.append(calculate_nu(min_slope, i))
        max_nus.append(calculate_nu(max_slope, i))
        nu_errs.append(abs(max_nus[i - 1] - min_nus[i - 1]))
        # print(f"{i} : {r2}")
        plt.plot(shifts, shifts * slope, alpha=0.8, linestyle="--", color=c)
    plt.xlabel(r"$z_0$")
    plt.ylabel(r"$z_{peak}$")
    # plt.savefig("./report/zpeaks.pdf")
    plt.close("peaks")

    plt.figure("nus")
    system_sizes = np.array([2**n for n in range(1, 9)])
    plt.errorbar(
        system_sizes,
        nus[:-1],
        yerr=nu_errs[:-1],
        marker="o",
        linestyle="none",
        capsize=2.5,
        color="m",
        alpha=0.9,
        label="Current work",
    )
    print(nus[-2], nu_errs[-2])
    print(nus)
    print(nus[:-1])
    plt.axhline(
        2.593, linestyle="--", color="b", alpha=0.5, label="Slevin & Ohtsuki 2009"
    )
    plt.axhline(2.51, linestyle="--", color="r", alpha=0.5, label="Roemer & Shaw 2025")
    plt.xticks([2, 8, 16, 32, 64, 128, 256])
    plt.xlabel(r"$2^n$")
    plt.ylabel(r"$\nu$")
    plt.legend()
    # plt.savefig("./report/nu.pdf")
    plt.close("nus")


def plot_t(
    t_folder: str, output_dir: str, start_step: int, end_step: int, version: str
):
    """
    Generate and save the distribution plot for t values across RG steps.

    Args:
        t_folder (str): Path to folder containing t histograms.
        output_dir (str): Directory to save output plot.
        start_step (int): First RG step to plot.
        end_step (int): Last RG step to plot.
        version (str): Version name to plot.

    Returns:
        None. Side effects: writes plot to output_dir.
    """
    output_filename = f"{output_dir}/t_distribution_{version}.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.set_title("Distribution of P(t)")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 3.0))
    starting = np.linspace(0, 1, 1000)
    ax.scatter(starting[::25], 2 * starting[::25], label="Initial dist", marker=".")
    k = 0
    for i in range(start_step, end_step):
        t_file = f"{t_folder}/input_t_hist_RG{i}.npz"
        data = np.load(t_file)
        hist = data["histval"]
        bins = data["binedges"]
        centers = data["bincenters"]
        density = get_density(hist, bins)
        if i in [start_step, end_step - 1]:
            ax.scatter(
                centers[::20],
                density[::20],
                label=f"RG step {i + 1}",
                marker=markers[k],
            )
            k += 1
    ax.legend(loc="upper left")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)


def plot_z(
    z_folder: str,
    output_dir: str,
    start_step: int,
    end_step: int,
    sym: bool,
    version: str,
    shift: float = 0.0,
):
    """
    Generate and save the distribution plot for z values across RG steps.

    Args:
        z_folder (str): Path to folder containing z histograms.
        output_dir (str): Directory to save output plot.
        start_step (int): First RG step to plot.
        end_step (int): Last RG step to plot.
        sym (bool): Whether to plot symmetrized z (True) or shifted (False).
        version (str): Version name to plot.
        shift (float, optional): Value of shift for unsymmetrized plots.

    Returns:
        None. Side effects: writes plot to output_dir.
    """
    if sym:
        output_filename = f"{output_dir}/sym_z_distribution_{version}.png"
        z_files = f"{z_folder}/sym_z_hist"
        title = "Distribution of Q(z)"
        y_bounds = (0.0, 0.25)
        x_bounds = (-5.0, 5.0)
    else:
        output_filename = f"{output_dir}/z_distribution_{version}_shift_{shift}.png"
        z_files = f"{z_folder}/z_hist_unsym"
        title = f"Distribution of Q(z - {shift})"
        y_bounds = (0.0, 0.25)
        x_bounds = (-5.0 + shift, 5.0 + shift)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("Q(z)")
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax1 = inset_locator.inset_axes(ax, width="30%", height="30%", loc="upper right")
    ax1.set_xlim((-25.0, 25.0))
    ax1.tick_params(labelright=True, labelleft=False)
    for i in range(start_step, end_step):
        z_file = f"{z_files}_RG{i}.npz"
        data = np.load(z_file)
        hist = data["histval"]
        bins = data["binedges"]
        centers = data["bincenters"]
        density = get_density(hist, bins)
        if i % 2 == 0:
            ax.scatter(centers[::100], density[::100], label=f"RG step {i}")
            ax1.plot(centers, density)
    ax.legend(loc="upper left")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = build_plot_parser()
    args = parser.parse_args()
    if os.path.exists(args.loc):
        config_path = build_config_path(args.loc, args.version, args.mode)
    else:
        config_path = str(config_file)
    config = load_yaml(config_path)
    print(f"Config loaded from {config_path}")
    rg_config = build_config(config)
    # Load constants
    version = str(args.version)
    num_rg = int(args.steps)
    fp_plot_dir = f"{data_dir}/{version}/FP/hist"
    exp_plot_dir = f"{data_dir}/{version}/EXP"
    t_fp_plot_dir = f"{fp_plot_dir}/input_t"
    z_fp_plot_dir = f"{fp_plot_dir}/sym_z"
    output_dir = "../report"
    os.makedirs(output_dir, exist_ok=True)
    start = 0
    end = num_rg
    plot_t(t_fp_plot_dir, output_dir, start, end, version)
    plot_z(z_fp_plot_dir, output_dir, start, end, True, version)
    for shift in SHIFTS:
        z_dir = f"{exp_plot_dir}/shift_{shift}/hist/z"
        plot_z(z_dir, output_dir, start, end, False, version, float(shift))

    # plot_z_fp()
    # plot_z_peaks()
