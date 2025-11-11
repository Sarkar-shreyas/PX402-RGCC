import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
from source.utilities import (
    l2_distance,
    hist_moments,
    get_density,
    STD_TOLERANCE,
    DIST_TOLERANCE,
)
import os
from collections import defaultdict
import json
from typing import Optional

DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm"
CURRENT_VERSION = "1.53S"
TYPE = "FP"
NUM_RG = 8


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
    filename: str,
    moments: defaultdict | list,
    distances: Optional[list] = None,
    rg_steps: int = NUM_RG,
):
    data = defaultdict(dict)
    if not distances:
        for i in range(rg_steps):
            mean = moments[i][1]
            std = moments[i][2]
            data[f"RG_{i}"]["mean"] = mean
            data[f"RG_{i}"]["std"] = std
            if i > 0:
                prev_std = moments[i - 1][2]
                if np.abs(std - prev_std) <= STD_TOLERANCE:
                    data[f"RG_{i}"]["std_converged"] = True
                else:
                    data[f"RG_{i}"]["std_converged"] = False
        for i in range(1, rg_steps):
            l2 = moments[i][0]
            data[f"RG_{i}"][f"L2 distance with RG_{i - 1}"] = l2
            if l2 <= DIST_TOLERANCE:
                data[f"RG_{i}"]["L2_converged"] = True
            else:
                data[f"RG_{i}"]["L2_converged"] = False
    else:
        for i in range(rg_steps):
            mean, std = moments[i]
            data[f"RG_{i}"]["mean"] = mean
            data[f"RG_{i}"]["std"] = std
            if i > 0:
                _, prev_std = moments[i - 1]
                if np.abs(std - prev_std) <= STD_TOLERANCE:
                    data[f"RG_{i}"]["std_converged"] = True
                else:
                    data[f"RG_{i}"]["std_converged"] = False
        for i in range(1, rg_steps):
            data[f"RG_{i}"][f"L2 distance with RG_{i - 1}"] = distances[i - 1]
            if distances[i - 1] <= DIST_TOLERANCE:
                data[f"RG_{i}"]["L2_converged"] = True
            else:
                data[f"RG_{i}"]["L2_converged"] = False

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Data has been saved to {filename}")


if __name__ == "__main__":
    # Load constants
    version = CURRENT_VERSION
    N = 80000000
    var_names = ["t", "g", "z", "input_t", "sym_z"]
    hist_dir = f"{DATA_DIR}/v{version}/{TYPE}/hist"
    stats_dir = f"{DATA_DIR}/v{version}/{TYPE}/stats"
    plots_dir = f"{DATA_DIR}/v{version}/{TYPE}/plots"
    t_folder = f"{hist_dir}/t"
    g_folder = f"{hist_dir}/g"
    z_folder = f"{hist_dir}/z"
    input_folder = f"{hist_dir}/input_t"
    sym_folder = f"{hist_dir}/sym_z"
    folder_names = {
        "hist": hist_dir,
        "stats": stats_dir,
        "plots": plots_dir,
        "t": t_folder,
        "g": g_folder,
        "z": z_folder,
        "input_t": input_folder,
        "sym_z": sym_folder,
    }
    for folder in folder_names:
        os.makedirs(folder_names[folder], exist_ok=True)
    print("Folders created")

    # Load histogram data
    map = defaultdict(list)
    for var in var_names:
        for i in range(NUM_RG):
            file = f"{folder_names[var]}/{var}_hist_RG{i}.npz"
            counts, bins, centers = load_hist_data(file)
            densities = get_density(counts, bins)
            map[var].append([counts, bins, centers, densities])
    print("All histograms have been loaded")

    # Plot the other 3 variables without clipping bounds
    other_vars = ["t", "g", "input_t"]
    for var in other_vars:
        plt.figure()
        plt.title(f"Histogram of dataset {var}")
        plt.xlabel(f"{var}")
        plt.ylabel(f"P({var})")
        for i in range(NUM_RG):
            plt.plot(map[var][i][2], map[var][i][3], label=f"RG{i}")
        plt.legend()
        plt.savefig(f"{plots_dir}/{var}_histogram.png", dpi=150)
    print("Plots for t, g and input t data have been made")

    # Store the moments calculated by the script
    moments = defaultdict(list)
    for i in range(NUM_RG):
        file = f"{stats_dir}/z_moments_RG{i}_{N}_samples.npz"
        l2, mean, std, converged = load_moments(file)
        moments[i] = [float(l2), float(mean), float(std), bool(converged)]
    print("Stats from the script have been stored")
    # print(moments)
    # print(moments)

    # Plot the unsymmetrised z data
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_xlim([-25.0, 25.0])
    ax0.set_ylim([0.0, 0.3])
    ax0.set_title("Unclipped histogram of Unsymmetrised z data")
    ax0.set_xlabel("z")
    ax0.set_ylabel("Q(z)")
    ax1.set_title("Clipped histogram of Unsymmetrised z data")
    ax1.set_xlabel("z")
    ax1.set_ylabel("Q(z)")
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([0, 0.3])
    z_dist = []
    z_moments = []
    for i in range(NUM_RG):
        if i > 0:
            z_dist.append(
                l2_distance(
                    map["z"][i][0],
                    map["z"][i - 1][0],
                    map["z"][i][1],
                    map["z"][i - 1][1],
                )
            )
        z_moments.append(hist_moments(map["z"][i][0], map["z"][i][1]))
        ax0.plot(map["z"][i][2], map["z"][i][3], label=f"RG{i}")
        ax1.plot(map["z"][i][2], map["z"][i][3], label=f"RG{i}")
    plt.legend()
    plt.savefig(f"{plots_dir}/z_histogram.png", dpi=150)
    print("Data plot made for unsymmetrised z")
    plt.close()

    # Plot the symmetrised z data
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))
    ax2.set_xlim([-5.0, 5.0])
    ax2.set_ylim([0.0, 0.3])
    ax2.set_title("Unclipped histogram of Symmetrised z data")
    ax2.set_xlabel("z")
    ax2.set_ylabel("Q(z)")
    ax3.set_title("Clipped histogram of Symmetrised z data")
    ax3.set_xlabel("z")
    ax3.set_ylabel("Q(z)")
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([0.16, 0.24])
    ax4 = inset_locator.inset_axes(ax2, width="30%", height=0.6)
    ax4.set_xlim([-25.0, 25.0])
    sym_z_dist = []
    sym_z_moments = []
    for i in range(NUM_RG):
        if i > 0:
            sym_z_dist.append(
                l2_distance(
                    map["sym_z"][i][0],
                    map["sym_z"][i - 1][0],
                    map["sym_z"][i][1],
                    map["sym_z"][i - 1][1],
                )
            )
        sym_z_moments.append(hist_moments(map["sym_z"][i][0], map["sym_z"][i][1]))
        ax2.plot(map["sym_z"][i][2], map["sym_z"][i][3], label=f"RG{i}")
        ax3.scatter(map["sym_z"][i][2][::50], map["sym_z"][i][3][::50], label=f"RG{i}")
        ax4.plot(map["sym_z"][i][2][::50], map["sym_z"][i][3][::50])
    ax2.legend(loc="upper left")
    ax3.legend()
    plt.savefig(f"{plots_dir}/sym_z_histogram.png", dpi=150)
    print("Data plot made for symmetrised z")

    # Manually calculate statistics to compare
    retrieved_file = f"{stats_dir}/retrieved_stats.json"
    z_file = f"{stats_dir}/z_stats.json"
    sym_z_file = f"{stats_dir}/sym_stats.json"
    construct_moments_dict(z_file, z_moments, z_dist)
    construct_moments_dict(sym_z_file, sym_z_moments, sym_z_dist)
    construct_moments_dict(
        retrieved_file,
        moments,
    )
    # print(map["sym_z"][0][1])
    print("Analysis done.")
