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

DATA_DIR = "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Data from taskfarm"


def load_hist_data(filename: str) -> list:
    data = np.load(filename)
    counts = data["histval"]
    bins = data["binedges"]
    centers = data["bincenters"]

    return [counts, bins, centers]


if __name__ == "__main__":
    version = 1
    var_names = ["t", "g", "z"]
    rgs = list(range(10))
    data = {}
    for var in var_names:
        for rg in rgs:
            filename = f"{DATA_DIR}/v{version}/data/hist/{var}_hist_RG{rg}.npz"
            sym_filename = f"{DATA_DIR}/v{version}/data/hist/{var}_hist_RG{rg}_sym.npz"

            # print(filename)
            if os.path.exists(filename):
                data[f"{var}_RG{rg}"] = load_hist_data(filename)
            if os.path.exists(sym_filename):
                data[f"{var}_RG{rg}_sym"] = load_hist_data(sym_filename)

    bins = data["t_RG0"][2]
    # print(data.keys())

    plt.title("z histogram")
    plt.xlabel("z")
    plt.ylabel("Q(z)")
    plt.xlim([-1, 3])
    plt.ylim([0.1, 0.3])
    for key in data.keys():
        a = get_density(data[key][0], data[key][1])
        if str(key)[0] == "z" and str(key)[-1] != "m":
            plt.plot(data[key][2], a, label=f"{key}")
    # n = [0, 2, 4, 8]
    # for i in n:
    #     filename = (
    #         f"{DATA_DIR}/v{version}/data/laundered t/t_laundered_RG{i}_batch_8.txt"
    #     )
    #     if os.path.exists(filename):
    #         input_t = np.loadtxt(filename)
    #         print("Loaded data")
    #         print(input_t.size)
    #         hist, _ = np.histogram(input_t, 1000, range=(0, 1), density=True)
    #         plt.plot(bins, hist, label=f"RG{i}")

    f = l2_distance(
        data["z_RG7"][0],
        data["z_RG8"][0],
        data["z_RG7"][1],
        data["z_RG8"][1],
    )
    mean_7, std_7 = hist_moments(data["z_RG7"][0], data["z_RG7"][1])
    mean_8, std_8 = hist_moments(data["z_RG8"][0], data["z_RG8"][1])
    print(f"L2 distance between unsymmetrised RG7&RG8: {f:.7f}")
    print(f"Unsymmetrised RG7 moments: Mean = {mean_7:.5f}, STD = {std_7:.7f}")
    print(f"Unsymmetrised RG8 moments: Mean = {mean_8:.5f}, STD = {std_8:.7f}")
    if f <= DIST_TOLERANCE:
        print("Unsymmetrised Dist Converged!")
    if np.abs(std_8 - std_7) <= STD_TOLERANCE:
        print("Unsymmetrised STD Converged!")
    print("=" * 100)
    f_sym = l2_distance(
        data["z_RG7_sym"][0],
        data["z_RG8_sym"][0],
        data["z_RG7_sym"][1],
        data["z_RG8_sym"][1],
    )
    mean_7_sym, std_7_sym = hist_moments(data["z_RG7_sym"][0], data["z_RG7_sym"][1])
    mean_8_sym, std_8_sym = hist_moments(data["z_RG8_sym"][0], data["z_RG8_sym"][1])
    print("=" * 100)
    print(f"L2 distance between symmetrised RG7&RG8: {f_sym:.7f}")
    print(f"Symmetrised RG7 moments: Mean = {mean_7_sym}, STD = {std_7_sym:.7f}")
    print(f"Symmetrised RG8 moments: Mean = {mean_8_sym}, STD = {std_8_sym:.7f}")
    if f_sym <= DIST_TOLERANCE:
        print("Symmetrised Dist Converged!")
    if np.abs(std_8_sym - std_7_sym) <= STD_TOLERANCE:
        print("Symmetrised STD Converged!")
    print("=" * 100)
    plt.legend()
    # plt.savefig(f"{DATA_DIR}/v{version}/data/plots/input_t_hist", dpi=150)
