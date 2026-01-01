import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
from source.utilities import get_density
from constants import CURRENT_VERSION, data_dir, NUM_RG, SHIFTS
import os

markers = ["*", "+", "d", "v", "s", "p"]


def plot_t(t_folder: str, output_dir: str, start_step: int, end_step: int):
    output_filename = f"{output_dir}/t_distribution_v{CURRENT_VERSION}.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.set_title("Distribution of P(t)")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 3.0))
    # ax0 = inset_locator.inset_axes(
    #     ax, width="30%", height="40%", loc="upper left", borderpad=1.5
    # )
    starting = np.linspace(0, 1, 1000)
    ax.scatter(starting[::25], 2 * starting[::25], label="Initial dist", marker=".")
    # ax0.set_xlim((0.4, 0.6))
    # ax0.set_ylim((0.7, 1.1))
    # ax0.tick_params(labelright=True, labelleft=False, labelbottom=True)
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
        # ax0.scatter(centers[::25], density[::25])

    ax.legend(loc="upper left")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)


def plot_z(
    z_folder: str,
    output_dir: str,
    start_step: int,
    end_step: int,
    sym: bool,
    shift: float = 0.0,
):
    if sym:
        output_filename = f"{output_dir}/sym_z_distribution_v{CURRENT_VERSION}.png"
        z_files = f"{z_folder}/sym_z_hist"
        title = "Distribution of Q(z)"
        y_bounds = (0.0, 0.25)
        x_bounds = (-5.0, 5.0)
    else:
        output_filename = (
            f"{output_dir}/z_distribution_v{CURRENT_VERSION}_shift_{shift}.png"
        )
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
    fp_plot_dir = f"{data_dir}/v{CURRENT_VERSION}/FP/hist"
    exp_plot_dir = f"{data_dir}/v{CURRENT_VERSION}/EXP"
    t_fp_plot_dir = f"{fp_plot_dir}/input_t"
    z_fp_plot_dir = f"{fp_plot_dir}/sym_z"
    output_dir = "../report"
    os.makedirs(output_dir, exist_ok=True)
    start = 0
    end = NUM_RG
    plot_t(t_fp_plot_dir, output_dir, start, end)
    plot_z(z_fp_plot_dir, output_dir, start, end, True)
    for shift in SHIFTS:
        z_dir = f"{exp_plot_dir}/shift_{shift}/hist/z"
        plot_z(z_dir, output_dir, start, end, False, float(shift))
