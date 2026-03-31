"""Critical exponent (ν) extraction and peak analysis for RG simulation outputs.

Post-processes FP and EXP histogram NPZ files produced by the RG Monte Carlo
pipeline to estimate the critical exponent ν that characterises the divergence of
the correlation length at the quantum phase transition (ξ ~ |δ|^{−ν}).

CLI usage
---------
::

    python -m analysis.critical_exponent \\
        --version VERSION --mode EXP --steps N [--loc local|taskfarm]

Expected input layout
---------------------
::

    {data_folder}/{version}/
    ├── FP/hist/sym_z/sym_z_hist_RG{steps-1}.npz   # symmetrised FP z-distribution
    └── EXP/shift{s}/hist/
        ├── z/z_hist_unsym_RG{i}.npz                # unsymmetrised EXP z-histograms
        └── {var}/{var}_hist_RG{i}.npz              # other variable histograms

Outputs written to disk
-----------------------
::

    {data_folder}/{version}/
    ├── peaks.json               — per-(RG step, shift) Gaussian peak estimates
    ├── overall_stats.json       — per-RG-step ν, slope, R² from peak and mean fits
    ├── z_peaks.png              — scatter/line plot of peak displacement vs shift
    ├── Nu_{N}_shifts.png        — ν vs system size (2^n) with error bars
    └── EXP/
        ├── stats/{shift}/       — per-shift moment statistics
        └── plots/{shift}/       — per-shift histogram PNG files

Stdout prints progress messages for each shift, wall-clock timing, and the final
paths of every file written.
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from source.utilities import calculate_nu, get_density, hist_moments, build_rng
from source.config import load_yaml, build_config
from source.fitters import estimate_z_peak, fit_z_peaks
from analysis.data_plotting import (
    load_hist_data,
    construct_moments_dict,
    plot_data,
    build_plot_parser,
    build_config_path,
)
import os
import json
from time import time
from constants import data_dir, config_file, local_dir

TYPE = "EXP"


def slice_middle(
    counts: np.ndarray,
    bins: np.ndarray,
    centers: np.ndarray,
    densities: np.ndarray,
    shift: float,
) -> tuple:
    """Extract the central region of a histogram within the window [-25+shift, 25+shift].

    Selects all bins whose centres fall inside the half-open interval
    ``[-25 + shift, 25 + shift]`` and returns consistent sub-arrays for
    counts, edges, centres, and densities so all four remain aligned.

    Args:
        counts: Raw histogram bin counts.  Shape ``(n_bins,)``.
        bins: Bin edge array.  Shape ``(n_bins + 1,)``; the returned slice
            includes one extra trailing edge so edges remain consistent with
            the returned ``centers``.
        centers: Centre of each bin.  Shape ``(n_bins,)``.
        densities: Normalised probability density for each bin (counts divided
            by bin width and total area).  Shape ``(n_bins,)``.
        shift: Scalar value that translates the extraction window; typically
            the EXP perturbation shift applied to the FP distribution.

    Returns:
        A 4-tuple ``(counts, bins, centers, densities)`` containing only the
        elements that fall within ``[-25 + shift, 25 + shift]``.  The
        ``bins`` array has one more element than the other three arrays.
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
    """Extract ν from FP and EXP histogram files and write results to disk.

    Order of operations:

    1. **Parse CLI args** — resolve ``--version``, ``--mode``, ``--steps``,
       and ``--loc`` (local vs taskfarm data folder).
    2. **Load config** — read the run YAML and build a typed
       :class:`~source.config.IQHEConfig` or :class:`~source.config.QSHEConfig`.
    3. **Load FP z-histogram** — read the symmetrised fixed-point distribution
       ``sym_z_hist_RG{steps-1}.npz`` which serves as the reference (shift = 0)
       for all peak-displacement measurements.
    4. **Load EXP histograms** — for each positive shift value and each
       observable variable, read all ``steps`` RG-step histogram files;
       generate histogram overlay plots and per-shift moment statistics.
    5. **Peak estimation** — for each (shift, RG step) pair, slice the z
       histogram to the central window via :func:`slice_middle` and call
       :func:`~source.fitters.estimate_z_peak` to obtain bootstrapped
       Gaussian mean estimates ``(min_peak, max_peak, overall_peak)``.
    6. **ν calculation** — for each RG step, subtract the shift=0 baseline
       from each peak array, call :func:`~source.fitters.fit_z_peaks` to
       obtain the linear slope of (shift, displacement), then pass the slope
       to :func:`~source.utilities.calculate_nu` to compute ν.
    7. **Write outputs** — serialise peak data and ν statistics to JSON and
       save diagnostic plots.

    Args:
        None.  Reads ``sys.argv`` via :func:`~analysis.data_plotting.build_plot_parser`.

    Returns:
        None.

    Side effects:
        Writes the following files (relative to ``{data_folder}/{version}/``):

        - ``peaks.json`` — per-(RG step, shift) Gaussian peak estimates::

            {
              "RG{i}": {
                "Peaks":       [float, ...],  // overall Gaussian means, one per shift
                "Min Peaks":   [float, ...],  // bootstrap lower bounds
                "Max Peaks":   [float, ...],  // bootstrap upper bounds
                "Peak Errors": [float, ...]   // max_peak - min_peak per shift
              }
            }

        - ``overall_stats.json`` — per-RG-step ν, slopes, and R²::

            {
              "RG{i}": {
                "Peak Nu":    float,  // ν from Gaussian-peak displacements
                "Mean Nu":    float,  // ν from histogram-moment means
                "Peak Slope": float,  // linear slope feeding into Peak Nu
                "Mean Slope": float,  // linear slope feeding into Mean Nu
                "Peak R2":    float,
                "Mean R2":    float
              }
            }

        - ``z_peaks.png`` — scatter and line-fit plot of peak displacement vs shift.
        - ``Nu_{N}_shifts.png`` — ν vs system size (2^n) with error bars.
        - ``EXP/plots/{shift}/{var}_hist_{shift}.png`` — histogram overlays per shift.
        - ``EXP/stats/{shift}/`` — moment statistics per shift.
    """
    parser = build_plot_parser()
    args = parser.parse_args()
    if args.loc is not None:
        config_path = build_config_path(args.loc, args.version, args.mode)
        print(config_path)
    else:
        config_path = str(config_file)
    config = load_yaml(config_path)
    rg_config = build_config(config)
    seed = rg_config.seed
    rng = build_rng(seed)
    sampler = rg_config.resample
    runtype = rg_config.type.upper()
    version = str(args.version)
    num_rg = int(args.steps)
    rg = num_rg + 1
    if str(args.loc).strip().lower() == "local":
        data_folder = local_dir
    else:
        data_folder = data_dir
    main_dir = f"{data_folder}/{version}"
    stats_dir = f"{data_folder}/{version}/{runtype}/stats"
    plots_dir = f"{data_folder}/{version}/{runtype}/plots"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    data_map = defaultdict(dict)
    if rg_config.model == "qshe":
        vars = rg_config.vars
    else:
        vars = ["t", "z", "input_t"]
    print(f"Performing peak estimation for {version}")
    print("=" * 100)
    # Load the FP distribution
    fp_file = (
        # symmetrised FP z-histogram at the last completed RG step
        f"{data_folder}/{version}/FP/hist/sym_z/sym_z_hist_RG{rg_config.steps - 1}.npz"
    )
    fp_counts, fp_bins, fp_centers = load_hist_data(fp_file)
    fp_density = get_density(fp_counts, fp_bins)
    shifts = np.array([shift for shift in rg_config.shifts if shift >= 0.0])
    # Get all the initial plots made for inspection
    start = time()
    for shift in shifts:
        for var in vars:
            data_map[shift][var] = []
            # EXP histogram directory: one subdirectory per shift value, per variable
            shift_dir = f"{data_folder}/{version}/{runtype}/shift{shift}/hist/{var}"
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
                    # unsymmetrised z: symmetrisation is applied later during peak estimation
                    filename = f"{shift_dir}/{var}_hist_unsym_RG{i - 1}.npz"
                else:
                    # all other variables use the standard (unmodified) histogram files
                    filename = f"{shift_dir}/{var}_hist_RG{i - 1}.npz"
                counts, bins, centers = load_hist_data(filename)
                densities = get_density(counts, bins)
                data_map[shift][var].append([counts, bins, centers, densities])
            filename = f"{shift_plot_dir}/{var}_hist_{shift}.png"
            plot_data(var, filename, data_map[shift][var], runtype, num_rg)
        construct_moments_dict(
            shift_stats_dir, shift_plot_dir, vars, data_map[shift], num_rg
        )

        print(f"Plots for shift {shift} have been made.")
        print(f"Stats for shift {shift} have been made.")
        print("-" * 100)
    print("=" * 100)
    # print(data_map.keys())
    fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_0.set_xlim([0, float(max(shifts))])
    # ax_0.set_ylim([0.0, 2])
    ax_0.set_title("Scatter plot of z peaks")
    ax_0.set_xlabel("z_0")
    ax_0.set_ylabel("z_peak")
    ax_1.set_title("Scatter plot and line fit of z peaks")
    ax_1.set_xlabel("z_0")
    ax_1.set_ylabel("z_peak")
    ax_1.set_xlim([0, float(max(shifts))])
    # ax_1.set_ylim([0, 2])

    peaks = np.zeros((rg, len(shifts))).astype(float)
    min_peaks = np.zeros((rg, len(shifts))).astype(float)
    max_peaks = np.zeros((rg, len(shifts))).astype(float)
    peak_errs = np.zeros((rg, len(shifts))).astype(float)
    means = np.zeros((rg, len(shifts))).astype(float)
    stds = np.zeros((rg, len(shifts))).astype(float)
    print("Beginning peak estimations")
    print("-" * 100)
    for j in range(len(shifts)):
        shift = shifts[j]
        shift_val = float(shift)
        print(f"Estimating peak for shift {shift}")
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
            min_peaks[i, j], max_peaks[i, j], peaks[i, j] = estimate_z_peak(
                sliced_counts, sliced_bins, sliced_centers, rng, sampler
            )
            peak_errs[i, j] = max_peaks[i, j] - min_peaks[i, j]
            means[i, j] = mean
            stds[i, j] = std
        print(f"Peak estimated for shift {shift} after {time() - start:.3f} seconds")
    print("Finished peaks estimation for every shift")
    print("=" * 100)
    # print(z_moments)
    overall_stats = defaultdict(dict)
    peak_data = defaultdict(dict)
    peak_data_file = f"{main_dir}/peaks.json"  # per-(RG step, shift) peak estimates for downstream inspection
    overall_stats_file = f"{main_dir}/overall_stats.json"  # primary output: ν estimates, slopes, and R² for every RG step
    x = np.array(shifts).astype(float)
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
        x_fit = x[:]
        y_fit = y[:]
        m_fit = m[:]
        ms, mr2 = fit_z_peaks(x_fit, m_fit)
        slope, r2 = fit_z_peaks(x_fit, y_fit)
        ax_0.set_title("Means")
        ax_1.set_title("Estimated peaks")
        if i in (1, 2, 3, 4, 5, 6, 7):
            ax_0.scatter(x_fit[1:], m_fit[1:])
            ax_0.plot(x, ms * x, label=f"RG_{i}")
            e = ax_1.errorbar(
                x_fit[1:],
                y_fit[1:],
                yerr=peak_errs[i, 1:],
                marker="o",
                linestyle="none",
                capsize=2.5,
            )
            c = e[0].get_color()
            x_line = np.linspace(0, float(max(shifts)) + shifts[1], 200)
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
    Nu_plot = f"{main_dir}/Nu_{len(shifts)}_shifts.png"
    plt.savefig(z_peaks_plot, dpi=150)
    plt.close()
    with open(overall_stats_file, "w") as f:
        json.dump(overall_stats, f, indent=2)
    with open(peak_data_file, "w") as f:
        json.dump(peak_data, f, indent=2)
    print(f"Overall stats for z saved to {overall_stats_file}")
    print(f"Peak data saved to {peak_data_file}")
    print(f"z peaks data plotted and saved to {z_peaks_plot}")
    system_size = [
        2**i for i in range(starting_index, rg)
    ]  # system size grows as 2^n with each RG step
    fig, (ax_2, ax_3) = plt.subplots(1, 2, figsize=(10, 4))
    ax_2.set_title("Scatter plot of Nu vs System size from means")
    ax_2.set_xlabel("2^n")
    ax_2.set_ylabel("Nu")
    ax_3.set_title("Scatter plot of Nu vs System size from peaks")
    ax_3.set_xlabel("2^n")
    ax_3.set_ylabel("Nu")
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

    plt.savefig(Nu_plot, dpi=150)
    plt.close()
    print(f"Nu data plotted and saved to {Nu_plot}")
    print("-" * 100)
    print("-" * 100)
    print(f"Analysis done after {time() - start:.3f} seconds")


if __name__ == "__main__":
    main()
