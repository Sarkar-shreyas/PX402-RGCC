import numpy as np
import matplotlib.pyplot as plt
from source.utilities import (
    l2_distance,
    mean_squared_distance,
    hist_moments,
    get_density,
    STD_TOLERANCE,
    DIST_TOLERANCE,
)
from source.fitters import std_derivative
import os
from collections import defaultdict
import json
from constants import (
    DATA_DIR,
    CURRENT_VERSION,
    NUM_RG,
    LEGENDS,
    XLIMS,
    YLIMS,
)

TYPE = "FP"


# ---------- Plotting helpers ---------- #
def plot_data(var: str, filename: str, data: list, mode: str):
    # plt.ylim((0.0, 1.01 * max(y_data)))

    xlim = XLIMS[mode][var]
    ylim = YLIMS[mode][var]
    legend_loc = LEGENDS[mode][var]
    if var == "z" or var == "sym_z":
        fig, (ax1, ax2) = plt.subplots(1, 2, num=f"{var}", figsize=(12, 4))
        ax1.set_title(f"Histogram of {var}")
        ax1.set_xlabel(f"{var}")
        ax1.set_ylabel(f"Q({var})")
        ax2.set_title(f"Clipped Histogram of {var}")
        ax2.set_xlabel(f"{var}")
        ax2.set_ylabel(f"Q({var})")

        ax1.set_xlim(xlim)
        ax2.set_xlim((-3.0, 3.0))
        ax2.set_ylim(ylim)
        # inset = inset_locator.inset_axes(ax, width="25%", height=1.0)
        # inset.set_xlim([-25.0, 25.0])
        for i in range(0, NUM_RG, 1):
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
        for i in range(NUM_RG):
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
    print(f"Figure plotted for {var} to {filename}")


def plot_moments(l2: list, moment_list: list[tuple], filename: str):
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
    rgs = [i + 1 for i in range(NUM_RG)]
    mean, std = map(np.array, zip(*moment_list))

    # mean_errors = std / np.sqrt(N)
    # std_errors = std / np.sqrt(2 * (N - 1))

    std_primes = std_derivative(rgs, std, 1)
    for i in range(NUM_RG):
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


# ---------- Stats helpers ---------- #
def load_hist_data(filename: str) -> tuple:
    data = np.load(filename)
    counts = data["histval"]
    bins = data["binedges"]
    centers = data["bincenters"]
    return (counts, bins, centers)


def load_moments(filename: str) -> tuple:
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
):
    data = defaultdict(dict)
    for var in vars:
        l2_dist = []
        moments = []
        msd = []
        for i in range(NUM_RG):
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
        plot_moments(l2_dist, moments, plot_file)
        stats_file = f"{stats_directory}/{var}_moments.json"
        with open(stats_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Stats for {var} has been saved to {stats_file}")


if __name__ == "__main__":
    # Load constants
    version = CURRENT_VERSION

    var_names = ["t", "z", "input_t", "sym_z"]
    z_vars = ["z", "sym_z"]
    other_vars = ["t", "input_t"]
    hist_dir = f"{DATA_DIR}/v{version}/{TYPE}/hist"
    stats_dir = f"{DATA_DIR}/v{version}/{TYPE}/stats"
    plots_dir = f"{DATA_DIR}/v{version}/{TYPE}/plots"
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
        for i in range(NUM_RG):
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
        plot_data(var, filename, data_map[var], TYPE)
    print("Plots for t, g and input t data have been made")
    print("-" * 100)
    construct_moments_dict(stats_dir, plots_dir, var_names, data_map)
    print("-" * 100)
    # print(data_map["sym_z"][0][1])
    print("Analysis done.")
