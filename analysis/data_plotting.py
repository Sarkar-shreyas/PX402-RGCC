import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from source.utilities import (
    l2_distance,
    mean_squared_distance,
    hist_moments,
    get_density,
)
from source.config import load_yaml, build_config
from source.fitters import std_derivative
import os
from collections import defaultdict
import json
from constants import (
    data_dir,
    local_dir,
    LEGENDS,
    XLIMS,
    YLIMS,
    config_file,
)
from mpl_toolkits.mplot3d import Axes3D

TYPE = "FP"
DIST_TOLERANCE = 1e-3
STD_TOLERANCE = 5e-4


# ---------- CLI helpers ---------- #
def build_plot_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for analysis scripts.

    Returns:
        argparse.ArgumentParser: Configured parser for CLI options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loc", default="remote", help="Location of the data, remote or local"
    )
    parser.add_argument("--version", required=True, help="Version name")
    parser.add_argument(
        "--mode",
        default="FP",
        choices=["FP", "EXP"],
        help="Mode of RG to plot, FP or EXP",
    )
    parser.add_argument(
        "--steps", default=9, help="Number of RG steps to use for analysis/plotting"
    )

    return parser


def build_config_path(data_location: str, version_name: str, rg_mode: str) -> str:
    """
    Construct the path to the config file for a given data location and version.

    Args:
        data_location (str): 'remote' or 'local'.
        version_name (str): Version string (folder name).
        rg_mode (str): 'FP' or 'EXP'.

    Returns:
        str: Path to the config YAML file.

    Notes:
        If the config file does not exist, scripts will fall back to legacy heuristics.
    """
    if data_location.strip().lower() == "remote":
        data_folder = data_dir
    elif data_location.strip().lower() == "local":
        data_folder = local_dir
    else:
        raise ValueError(
            f"Invalid data location {data_location} entered. Expected 'remote' or 'local'."
        )
    version = version_name.strip().lower()
    version = f"{version[:-1]}{version[-1].upper()}"
    rg_mode = rg_mode.strip().upper()
    config_location = f"{data_folder}/{version}/{rg_mode}/updated_config.yaml"

    return config_location


# ---------- Plotting helpers ---------- #
def plot_3d(hist_2d_filename: str, num_rg: int, output_filename: str):
    # hist_data = np.load(hist_2d_filename)
    fig, ax = plt.subplots()
    densities = []
    z_centers = None
    f_centers = None
    for i in range(num_rg):
        hist_data = np.load(f"{hist_2d_filename}_RG{i}.npz")
        densities.append(hist_data["zfdensities"])

        if z_centers is None:
            z_centers = hist_data["zcenters"]

        if f_centers is None:
            f_centers = hist_data["fcenters"]

    print("Loaded all 2dhist files")

    densities = np.array(densities)
    steps, zshape, fshape = densities.shape

    zgrid, fgrid = np.meshgrid(z_centers[::50], f_centers[::50], indexing="ij")  # type: ignore
    vmin = densities.min()
    vmax = densities.max()
    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    cmap = plt.cm._colormaps["viridis"]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(num_rg):
        density = densities[i][::50]
        z_height = np.full_like(zgrid, fill_value=i, dtype=float)
        colors = cmap(norm(density))

        ax.plot_surface(
            zgrid,
            fgrid,
            z_height,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.9,
        )
        print(f"Figure plotted for RG {i}")
    ax.set_xlabel("z")
    # ax.set_xlim((-10.0, 10.0))
    ax.set_ylabel("f")
    ax.set_zlabel("RG step")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Histogram density")
    print("3D plot with colorbar generated.")
    plt.tight_layout()
    plt.savefig(output_filename)


def plot_data(var: str, filename: str, data: list, mode: str, num_rg: int):
    """
    Plot histogram or scatter data for a given variable and save to file.

    Args:
        var (str): Variable name ('t', 'z', etc.).
        filename (str): Output filename for the plot.
        data (list): Data to plot (see code for structure).
        mode (str): RG mode ('FP' or 'EXP').
        num_rg (int): Number of RG steps.

    Returns:
        None. Side effect: writes plot to file.

    Notes:
        Output file is written as PNG. Uses axis limits and legends from constants.py.
    """
    # plt.ylim((0.0, 1.01 * max(y_data)))

    xlim = XLIMS[mode][var]
    ylim = YLIMS[mode][var]
    legend_loc = LEGENDS[mode][var]
    if var == "z" or var == "sym_z":
        fig, (ax1, ax2) = plt.subplots(1, 2, num=f"{var}", figsize=(12, 8))
        ax1.set_title(f"Histogram of {var}")
        ax1.set_xlabel(f"{var}")
        ax1.set_ylabel(f"Q({var})")
        ax2.set_title(f"Clipped Histogram of {var}")
        ax2.set_xlabel(f"{var}")
        ax2.set_ylabel(f"Q({var})")

        ax1.set_xlim(xlim)
        ax2.set_xlim((-5.0, 5.0))
        # ax2.set_ylim((0.15, 0.25))
        # inset = inset_locator.inset_axes(ax, width="25%", height=1.0)
        # inset.set_xlim([-25.0, 25.0])
        for i in range(0, num_rg, 1):
            x_data = data[i][2]
            y_data = data[i][3]
            ax1.plot(x_data, y_data, label=f"RG{i}")
            ax2.scatter(x_data[::100], y_data[::100], label=f"RG{i}")
            ax1.legend(loc=legend_loc)
            ax2.legend(loc=legend_loc)
            # inset.plot(x_data, y_data)
    else:
        fig, ax = plt.subplots(num=f"{var}", figsize=(10, 10))
        ax.set_title(f"Histogram of {var}")
        ax.set_xlabel(f"{var}")
        ax.set_ylabel(f"P({var})")
        for i in range(num_rg):
            x_data = data[i][2]
            y_data = data[i][3]
            ax.plot(x_data, y_data, label=f"RG{i}")
        if var == "input_t":
            g = np.linspace(0, 1, 1000000)
            t = np.sqrt(g)
            h, b = np.histogram(t, 1000, (0, 1), density=True)
            ax.plot(x_data, h, label="Initial")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc=legend_loc)

    fig.savefig(filename, dpi=150)
    plt.close(fig)
    # print(f"Figure plotted for {var} to {filename}")


def plot_moments(l2: list, moment_list: list[tuple], filename: str, num_rg: int):
    """
    Plot moments (mean, std, derivative, MSD) over RG steps and save to file.

    Args:
        l2 (list): Mean squared distances per RG step.
        moment_list (list of tuple): List of (mean, std, std_derivative) per RG step.
        filename (str): Output filename for the plot.
        num_rg (int): Number of RG steps.

    Returns:
        None. Side effect: writes plot to file.

    Notes:
        Output file is written as PNG. Plots four subplots for moments.
    """
    moment_fig, ((moment_ax0, moment_ax1), (moment_ax2, moment_ax3)) = plt.subplots(
        2, 2, figsize=(10, 10)
    )
    moment_ax0.set_title("Mean over RG steps")
    moment_ax1.set_title("Standard deviation over RG steps")
    moment_ax2.set_title("Derivative of standard deviation over RG steps")
    moment_ax3.set_title("Mean Squared distance between RG steps")
    moment_ax0.set_xlabel("RG step")
    moment_ax1.set_xlabel("RG step")
    moment_ax2.set_xlabel("RG step")
    moment_ax3.set_xlabel("RG step")
    moment_ax0.set_ylabel("Mean")
    moment_ax1.set_ylabel("Standard Deviation")
    moment_ax2.set_ylabel("Derivative of STD")
    moment_ax3.set_ylabel("Mean Squared Distance")
    # moment_ax1.set_ylim([2.0, 2.2])
    # moment_ax2.set_ylim([0.0, 0.04])
    # moment_ax3.set_ylim([0.0, 0.0015])
    rgs = [i + 1 for i in range(num_rg)]
    mean, std = map(np.array, zip(*moment_list))

    # mean_errors = std / np.sqrt(N)
    # std_errors = std / np.sqrt(2 * (N - 1))

    std_primes = std_derivative(rgs, std, 1)
    for i in range(num_rg):
        moment_ax0.scatter(i, mean[i])
        moment_ax1.scatter(i, std[i])
        moment_ax2.scatter(i, std_primes[i])
        # moment_ax0.errorbar(i, mean[i], yerr=mean_errors[i], fmt="o")
        # moment_ax1.errorbar(i, std[i], yerr=std_errors[i], fmt="o")
        if i > 0:
            moment_ax3.scatter(i + 1, l2[i - 1])
    moment_fig.tight_layout()
    moment_fig.savefig(filename, dpi=150)
    plt.close(moment_fig)


def calculate_average_nu(
    data: dict,
    starting_step: int,
    rg_steps: int,
):
    """
    Calculate and print the average critical exponent Nu and its error over a range of RG steps.

    Args:
        data (dict): Dictionary of stats loaded from JSON (keys: RG step, values: dicts with 'Peak Nu', 'Peak Slope', 'Peak R2').
        starting_step (int): First RG step to include.
        rg_steps (int): Last RG step (exclusive).

    Returns:
        None. Prints average Nu and error to stdout.

    Notes:
        - Assumes data is structured as in new-format output JSON.
        - For legacy data, may be missing keys or have different structure (assumption: code will error if so).
    """
    loaded_data = defaultdict(list)
    for i in range(starting_step, rg_steps):
        loaded_data["Nu"].append(data[f"RG{i}"]["Peak Nu"])
        # loaded_data["Nu error"].append(errors[i - 1])
        loaded_data["R2"].append(data[f"RG{i}"]["Peak R2"])
        print(
            f"At step {i}, Nu = {data[f'RG{i}']['Peak Nu']:.5f}, with slope = {data[f'RG{i}']['Peak Slope']:.5f} and R2 = {data[f'RG{i}']['Peak R2']:.5f}"
        )
    avg_error = np.max(loaded_data["Nu"][:]) - np.min(loaded_data["Nu"][:])
    avg_nu = np.mean(loaded_data["Nu"][:])
    print(
        f"Average Nu value from RG steps {starting_step}-{rg_steps - 1} = {avg_nu:.5f} \u00b1 {avg_error:.5f}"
    )


# ---------- Stats helpers ---------- #
def load_hist_data(filename: str) -> tuple:
    """
    Load histogram data from a .npz file.

    Args:
        filename (str): Path to the .npz file containing 'histval', 'binedges', 'bincenters'.

    Returns:
        tuple: (counts, bins, centers) arrays from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required arrays are missing.
    """
    data = np.load(filename)
    try:
        counts = data["histval"]
    except KeyError:
        print(data)
        counts = data["counts"]
    bins = data["binedges"]
    centers = data["bincenters"]
    return (counts, bins, centers)


def load_moments(filename: str) -> tuple:
    """
    Load moment statistics from a .npz file.

    Args:
        filename (str): Path to the .npz file containing 'l2_dist', 'mean', 'std', 'converged'.

    Returns:
        tuple: (l2, mean, std, conv) arrays from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required arrays are missing.
    """
    data = np.load(filename)
    l2 = data["l2_dist"]
    mean = data["mean"]
    std = data["std"]
    conv = data["converged"]
    return (l2, mean, std, conv)


def construct_moments_dict(
    stats_directory: str,
    plots_directory: str,
    vars: list,
    data_map: dict | defaultdict,
    num_rg: int,
):
    """
    Construct and save moment statistics and plots for each variable over RG steps.

    Args:
        stats_directory (str): Directory to save JSON stats files.
        plots_directory (str): Directory to save PNG plots.
        vars (list): List of variable names to process.
        data_map (dict or defaultdict): Mapping from variable name to list of histogram data per RG step.
        num_rg (int): Number of RG steps.

    Returns:
        None. Side effects: writes JSON stats and PNG plots to disk.

    Notes:
        - Output files are named <var>_moments.json and <var>_moments.png.
        - Assumes data_map is populated for all vars and steps.
    """
    data = defaultdict(dict)
    for var in vars:
        l2_dist = []
        moments = []
        msd = []
        for i in range(num_rg):
            counts = data_map[var][i][0]
            bins = data_map[var][i][1]
            if i > 0:
                prev_counts = data_map[var][i - 1][0]
                prev_bins = data_map[var][i - 1][1]
                l2 = l2_distance(counts, prev_counts, bins, prev_bins)
                l2_dist.append(l2)
                mean_square_dist = mean_squared_distance(
                    prev_counts, counts, prev_bins, bins
                )
                msd.append(mean_square_dist)
                data[f"RG_{i}"][f"L2 distance with RG_{i - 1}"] = l2_dist[i - 1]
                data[f"RG_{i}"][f"Mean squared distance with RG_{i - 1}"] = msd[i - 1]
                if l2_dist[i - 1] <= DIST_TOLERANCE:
                    data[f"RG_{i}"]["L2_converged"] = True
                else:
                    data[f"RG_{i}"]["L2_converged"] = False
                if msd[i - 1] <= DIST_TOLERANCE:
                    data[f"RG_{i}"]["MSD converged"] = True
                else:
                    data[f"RG_{i}"]["MSD converged"] = False
            mean, std = hist_moments(counts, bins)
            moments.append((mean, std))
            data[f"RG_{i}"]["mean"] = mean
            data[f"RG_{i}"]["std"] = std
            if i > 0:
                _, prev_std = moments[i - 1]
                if np.abs(std - prev_std) <= STD_TOLERANCE:
                    data[f"RG_{i}"]["std_converged"] = True
                else:
                    data[f"RG_{i}"]["std_converged"] = False
        plot_file = f"{plots_directory}/{var}_moments.png"
        plot_moments(l2_dist, moments, plot_file, num_rg)
        stats_file = f"{stats_directory}/{var}_moments.json"
        with open(stats_file, "w") as f:
            json.dump(data, f, indent=2)
        # print(f"Stats for {var} has been saved to {stats_file}")


if __name__ == "__main__":
    parser = build_plot_parser()
    args = parser.parse_args()
    if os.path.exists(args.loc):
        config_path = build_config_path(args.loc, args.version, args.mode)
        if str(args.loc).strip().lower() == "remote":
            data_folder = data_dir
        elif str(args.loc).strip().lower() == "local":
            data_folder = local_dir
    else:
        config_path = str(config_file)
        data_folder = data_dir
    config = load_yaml(config_path)
    print(f"Config loaded from {config_path}")
    rg_config = build_config(config)
    # Load constants

    version = str(args.version)
    num_rg = int(args.steps)
    var_names = ["t", "z", "input_t", "sym_z"]
    z_vars = ["z", "sym_z"]
    other_vars = ["t", "input_t"]
    hist_dir = f"{data_folder}/{version}/{TYPE}/hist"
    stats_dir = f"{data_folder}/{version}/{TYPE}/stats"
    plots_dir = f"{data_folder}/{version}/{TYPE}/plots"
    t_folder = f"{hist_dir}/t"
    z_folder = f"{hist_dir}/z"
    input_folder = f"{hist_dir}/input_t"
    sym_folder = f"{hist_dir}/sym_z"
    folder_names = {
        "hist": hist_dir,
        "stats": stats_dir,
        "plots": plots_dir,
        "t": t_folder,
        "z": z_folder,
        "input_t": input_folder,
        "sym_z": sym_folder,
    }
    for folder in folder_names:
        os.makedirs(folder_names[folder], exist_ok=True)
    print("Folders created")

    # Load histogram data
    data_map = defaultdict(list)
    for var in var_names:
        for i in range(num_rg):
            if var == "z":
                file = f"{folder_names[var]}/{var}_hist_unsym_RG{i}.npz"
            else:
                file = f"{folder_names[var]}/{var}_hist_RG{i}.npz"
            counts, bins, centers = load_hist_data(file)
            densities = get_density(counts, bins)
            data_map[var].append([counts, bins, centers, densities])
    print("All histogram datasets have been loaded")

    # Plot the other 3 variables without clipping bounds
    for var in var_names:
        filename = f"{plots_dir}/{var}_histogram.png"
        plot_data(var, filename, data_map[var], TYPE, num_rg)
    print("Plots for t, z and input t data have been made")
    print("-" * 100)
    construct_moments_dict(stats_dir, plots_dir, var_names, data_map, num_rg)
    # print("-" * 100)
    # print(data_map["sym_z"][0][1])
    print("Analysis done.")
