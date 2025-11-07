import numpy as np
import matplotlib.pyplot as plt
from Taskfarm.source.utilities import get_density, l2_distance, hist_moments
import os
from pathlib import Path

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

    # print(data.keys())

    plt.title("z histogram, truncated")
    plt.xlabel("z")
    plt.ylabel("Q(z)")
    plt.xlim([-1, 3])
    plt.ylim([0.1, 0.3])
    for key in data.keys():
        a = get_density(data[key][0], data[key][1])
        if str(key)[0] == "z" and str(key)[-1] != "m":
            plt.plot(data[key][2], a, label=f"{key}")

    f = l2_distance(
        data["z_RG7"][0],
        data["z_RG8"][0],
        data["z_RG7"][1],
        data["z_RG8"][1],
    )
    mean_7, std_7 = hist_moments(data["z_RG7"][0], data["z_RG7"][1])
    mean_8, std_8 = hist_moments(data["z_RG8"][0], data["z_RG8"][1])
    print(f"L2 distance between unsymmetrised RG7&RG8: {f:.9f}")
    print(f"Unsymmetrised RG7 moments: Mean = {mean_7:.5f}, STD = {std_7:.7f}")
    print(f"Unsymmetrised RG8 moments: Mean = {mean_8:.5f}, STD = {std_8:.7f}")
    print("=" * 100)
    f_sym = l2_distance(
        data["z_RG7_sym"][0],
        data["z_RG8_sym"][0],
        data["z_RG7_sym"][1],
        data["z_RG8_sym"][1],
    )
    mean_7_sym, std_7_sym = hist_moments(data["z_RG7_sym"][0], data["z_RG7_sym"][1])
    mean_8_sym, std_8_sym = hist_moments(data["z_RG8_sym"][0], data["z_RG8_sym"][1])
    print(f"L2 distance between symmetrised RG7&RG8: {f_sym:.9f}")
    print(f"Symmetrised RG7 moments: Mean = {mean_7_sym}, STD = {std_7_sym:.7f}")
    print(f"Symmetrised RG8 moments: Mean = {mean_8_sym}, STD = {std_8_sym:.7f}")
    plt.legend()
    plt.savefig(f"{DATA_DIR}/v{version}/data/plots/z_hist_truncated", dpi=150)
