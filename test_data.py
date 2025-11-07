import numpy as np
import matplotlib.pyplot as plt
from Taskfarm.source.utilities import (
    l2_distance,
    hist_moments,
    get_density,
    STD_TOLERANCE,
    DIST_TOLERANCE,
)
import os
import sys
from collections import defaultdict

DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Data from taskfarm"


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


if __name__ == "__main__":
    version = 1.1
    N = 120000000
    var_names = ["t", "g", "z", "input_t"]
    hist_dir = f"{DATA_DIR}/v{version}/hist"
    stats_dir = f"{DATA_DIR}/v{version}/stats"
    plots_dir = f"{DATA_DIR}/v{version}/plots"
    t_folder = f"{hist_dir}/t"
    g_folder = f"{hist_dir}/g"
    z_folder = f"{hist_dir}/z"
    input_folder = f"{hist_dir}/input"
    sym_folder = f"{DATA_DIR}/v{version}/hist/sym"
    folder_names = {
        "t": t_folder,
        "g": g_folder,
        "z": z_folder,
        "input_t": input_folder,
        "sym": sym_folder,
    }
    num_steps = 8
    map = defaultdict(list)
    for var in var_names:
        for i in range(num_steps):
            file = f"{folder_names[var]}/{var}_hist_RG{i}.npz"
            counts, bins, centers = load_hist_data(file)
            densities = get_density(counts, bins)
            map[var].append([densities, bins, centers])

    for i in range(num_steps):
        file = f"{folder_names['sym']}/z_hist_RG{i}_sym.npz"
        counts, bins, centers = load_hist_data(file)
        densities = get_density(counts, bins)
        map["sym"].append([densities, bins, centers])

    # print(len(map["t"][0]))
    # for key in map:
    #     plt.figure()
    #     plt.title(f"Histogram of dataset {key}")
    #     plt.xlabel(f"{key}")
    #     plt.ylabel(f"P({key})")
    #     for i in range(num_steps):
    #         plt.plot(map[key][i][1], map[key][i][0], label=f"RG{i}")
    #     plt.legend()
    #     plt.savefig(f"{DATA_DIR}/v{version}/plots/{key}_histogram", dpi=150)

    moments = defaultdict(list)
    for i in range(num_steps):
        file = f"{stats_dir}/z_moments_RG{i}_{N}_samples.npz"
        l2, mean, std, converged = load_moments(file)
        moments[i].append([l2, mean, std, converged])

    # print(moments)
    plt.figure()
    plt.title("Histogram of Unymmetrised z data")
    plt.xlabel("z")
    plt.ylabel("Q(z)")
    plt.xlim([-2, 4])
    plt.ylim([0, 0.3])
    dist = []
    moments = []
    for i in range(num_steps):
        if i > 0:
            dist.append(
                l2_distance(
                    map["z"][i][0],
                    map["z"][i - 1][0],
                    map["z"][i][1],
                    map["z"][i - 1][1],
                )
            )
        moments.append(hist_moments(map["z"][i][0], map["z"][i][1]))
        plt.plot(map["z"][i][2], map["z"][i][0], label=f"RG{i}")
    plt.legend()
    plt.savefig(f"{plots_dir}/z_truncated.png", dpi=150)

    for i in range(num_steps - 1):
        print(f"L2 distance between RG{i} and RG{i + 1} = {dist[i]}")
        if dist[i] <= DIST_TOLERANCE:
            print(f"The histogram converged for L2 distance by RG step {i + 1}!")

    for i in range(num_steps):
        mean, std = moments[i]
        print(f"Mean and STD for RG{i}: Mean = {mean:.7f}, STD = {std:.7f}")
        if i > 0:
            _, prev_std = moments[i - 1]
            if np.abs(std - prev_std) <= STD_TOLERANCE:
                print(f"The histogram converged for STD by RG step {i}")
