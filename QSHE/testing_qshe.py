# flake8: noqa: E501
from collections import defaultdict
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from source.utilities import (
    convert_t_to_z,
    convert_z_to_t,
    generate_constant_array,
    generate_initial_t_distribution,
    generate_random_phases,
    extract_t_samples,
    get_current_date,
    get_density,
    build_rng,
    hist_moments,
    rejection_sampler_2d,
    solve_qshe_matrix,
    build_2d_hist,
    inverse_cdf_2d,
    conditional_2d_resampler,
    # launder,
)
from numpy.typing import ArrayLike
from source.config import build_config, handle_config
from source.parse_config import build_parser
from time import time
import psutil
import os
import sys
from constants import T_DICT, qshe_dir, data_dir, PHI_DICT
import argparse

process = psutil.Process(os.getpid())


def get_memory_usage(item: str = "") -> None:
    """Prints the memory usage for a process"""
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"{item} = {memory:.3f} MB")


def numerical_solver(
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    N: int,
    output_index: int,
    inputs: ArrayLike,
    batch_size: int,
) -> np.ndarray:
    """Solve the matrix equation Mx=b for N samples using batching"""
    start = time()
    num_batches = N // batch_size
    output = np.empty(shape=(N, 1), dtype=np.float64)
    print(
        f"Beginning numerical solver for index {output_index} on {get_current_date()}"
    )
    print(f"Computing {num_batches} batches of size {batch_size}")
    # get_memory_usage("Memory usage before computation")
    for i in range(num_batches):
        indexes = slice(i * batch_size, (i + 1) * batch_size)
        output[indexes] = np.abs(
            solve_qshe_matrix(
                ts[indexes],
                fs[indexes],
                phis[indexes],
                batch_size,
                output_index,
                inputs,
            )
        )
        if (i + 1) in {10, 50, 100, num_batches}:
            # get_memory_usage(f"Memory usage after batch {i + 1}")
            print(f"Batch {i + 1} done in {time() - start:.3f} seconds")
    print(
        f"Computation for all {num_batches} batches of index {output_index} done after {time() - start:.3f} seconds"
    )
    print("-" * 100)
    return np.abs(output)


def check_single_node():
    """Performs a unitarity check for the single node S matrix"""
    f_val = np.random.random(size=1)
    split = 1 - f_val**2
    t = np.random.uniform(0, np.sqrt(split), 1000)
    r = np.sqrt(split - t**2)
    f = np.full(shape=1000, fill_value=f_val)
    # print(f, t[0], r[0])
    # print(max(np.abs(f) ** 2 + np.abs(t[0]) ** 2 + np.abs(r[0]) ** 2))
    var_sum = np.abs(f_val) ** 2 + np.abs(t[0]) ** 2 + np.abs(r[0]) ** 2
    try:
        assert np.abs(var_sum - 1) <= 1e-12
    except AssertionError:
        print(f"The sum is not smaller than or equal to 1 : {var_sum}")
        print(f"f = {np.abs(f_val) ** 2}")
        print(f"t = {np.abs(t[0]) ** 2}")
        print(f"r = {np.abs(r[0]) ** 2}")
        sys.exit(0)

    S = np.zeros(shape=(1000, 4, 4), dtype=np.complex128)
    S[:, 0, 0] = t
    S[:, 0, 1] = r
    S[:, 0, 3] = f
    S[:, 1, 1] = -t
    S[:, 1, 0] = r
    S[:, 1, 2] = -f
    S[:, 2, 2] = t
    S[:, 2, 3] = r
    S[:, 2, 1] = -f
    S[:, 3, 3] = -t
    S[:, 3, 2] = r
    S[:, 3, 0] = f

    S_transpose = np.conjugate(np.transpose(S, (0, 2, 1)))
    id_matrix = np.eye(4, dtype=np.complex128)[None, :, :]

    projection = S_transpose @ S
    error = np.max(np.abs(projection - id_matrix))
    print(f"Max error = {error}")
    assert np.allclose(id_matrix, projection, atol=1e-12, rtol=0)
    print("The S matrix is unitary")
    # sys.exit(0)


def append_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--t",
        type=int,
        default=0,
        help="Use a constant value for the t array. 1: t=0, 2: t=1/2, 3: t=1/sqrt(2), 4: t=1",
    )
    parser.add_argument(
        "--phi",
        type=int,
        default=0,
        help="Enter a constant value for the phi array. 1: phi=0, 2: phi=pi/4, 3: phi=pi/2, 4: phi=pi, 5: phi=2pi",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=0.0,
        help="Enter a constant float value for the f array.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enter command to plot all outputs, else uses default list in loaded config.",
    )

    return parser


def gen_initial_data(
    samples: int,
    t_val: int,
    phi_val: int,
    f_val: float,
    rng: np.random.Generator,
    fp_data: Optional[ArrayLike] = None,
) -> dict:
    """Generates initial t, phi and f arrays based on given inputs resolved from CLI input and config file parsing"""
    n = samples
    if f_val > 1.0:
        f_val = 1.0
    elif f_val < 0.0 or f_val < 1e-10:
        f_val = 0.0
    f_array = generate_constant_array(n, f_val, 5)
    if t_val == 0:
        # t_sample = launder(
        #     n,
        #     fp_data["histval"],
        #     fp_data["binedges"],
        #     fp_data["bincenters"],
        #     rng,
        #     config.resample,
        # )
        # t_array = extract_t_samples(t_sample, n, rng)
        split = 1 - f_array**2
        # t_array = rng.uniform(0, np.sqrt(split), size=(n, 5))
        t_sample = generate_initial_t_distribution(n, rng, split[0, 0])
        t_array = extract_t_samples(t_sample, n, rng)
    else:
        t_array = generate_constant_array(n, T_DICT[f"{t_val}"], 5)
    if phi_val == 0:
        phi_array = generate_random_phases(n, rng, 16)
    else:
        phi_array = generate_constant_array(n, PHI_DICT[f"{phi_val}"], 16)
    # split_array = np.full(shape=(n, 1), fill_value=split)

    data_dict = {"t": t_array, "f": f_array, "phi": phi_array, "split": split}
    return data_dict


def plot_2d_hist(data_dict: dict) -> None:
    """Generate various plots from 2d histogram data"""
    vars = ["z", "f"]
    for var in vars:
        # print(np.mean(data_dict[var]["densities"]), np.std(data_dict[var]["densities"]))
        plt.figure(var)
        plt.title(f"Distribution of {var}")
        # plt.xlim((-1.0, 1.0))
        plt.plot(data_dict[var]["bincenters"], data_dict[var]["densities"])
        plt.savefig(f"{var}_test.png", dpi=150)
        plt.close(var)
        print(f"Plot for {var} created")
    fig, ax = plt.subplots()
    zf_counts = data_dict["zf"]["counts"].T
    image = ax.imshow(
        zf_counts,
        origin="lower",
        extent=(
            data_dict["z"]["binedges"][0],
            data_dict["z"]["binedges"][-1],
            data_dict["f"]["binedges"][0],
            data_dict["f"]["binedges"][-1],
        ),
        aspect="auto",
        # norm=LogNorm(
        #     vmin=max(zf_counts.min(), 1e-6),
        #     vmax=zf_counts.max(),
        # ),
    )
    ax.set_xlabel("z")
    ax.set_ylabel("f")
    ax.set_xlim((-7.0, 7.0))
    fig.colorbar(image, ax=ax, label="p(z,f)")
    fig.savefig("zf_test.png", dpi=150)
    print("Plot for zf created")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Program started on {get_current_date()}")
    start_time = time()

    # Input parsing and config setup
    base_parser = build_parser()
    parser = append_parser(base_parser)
    args = parser.parse_args()
    config = handle_config(args.config, args.override)
    rg_config = build_config(config)
    n = rg_config.samples
    inputs = [float(i) for i in rg_config.inputs]
    num_phases = 16
    rng = build_rng(rg_config.seed)
    # check_single_node()

    print(
        f"Beginning QSHE RG workflow for {rg_config.steps} steps and {rg_config.samples} samples"
    )
    print(f"t = {T_DICT[str(args.t)]}, phi = {PHI_DICT[str(args.phi)]}, f = {args.f}")
    print(f"Inputs = {rg_config.inputs}")
    print(f"Outputs = {rg_config.outputs}")
    print("-" * 100)

    # Folder setup
    original_output = sys.stdout
    output_folder = f"{qshe_dir}/outputs"
    plots_folder = f"{qshe_dir}/plots/{get_current_date('day')}"
    day_output_folder = f"{output_folder}/{get_current_date('day')}"
    output_file_name = f"{day_output_folder}/qshe_{n}_outputs_f_{args.f}.txt"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(day_output_folder, exist_ok=True)
    print(f"Output being printed to {output_file_name}")

    output_file = open(output_file_name, "w")
    sys.stdout = output_file
    print(f"Program started on {get_current_date()}")
    print(f"Starting numerical solver for QSHE matrix with {n} samples")
    print("-" * 100)

    # Load latest FP distribution
    fp_file = f"{data_dir}/v1.90S/FP/hist/t/t_hist_RG{9}.npz"
    fp_data = np.load(fp_file)
    fp_hist = fp_data["histval"]
    fp_bins = fp_data["binedges"]
    fp_centers = fp_data["bincenters"]
    fp_density = get_density(fp_hist, fp_bins)

    # Generate initial arrays
    initial_data = gen_initial_data(
        rg_config.samples, args.t, args.phi, args.f, rng, fp_data
    )
    starting_t = initial_data["t"]
    starting_f = initial_data["f"]
    starting_phases = initial_data["phi"]
    split = initial_data["split"]
    print(
        f"Initial t, f and phi arrays generated after {time() - start_time:.3f} seconds"
    )
    print("-" * 100)

    # Set up initial plots
    plt.figure(num="outputs", figsize=(12, 6))
    plt.title(
        f"Distribution of outputs for f = {args.f} and phi = {PHI_DICT[str(args.phi)]}"
    )
    plt.xlabel("values")
    plt.ylabel("P(output)")
    plt.ylim((0, 10))
    starting_hist, starting_bins = np.histogram(
        starting_t, rg_config.t_bins, rg_config.t_range, density=True
    )
    plt.plot(
        (0.5 * (starting_bins[1:] + starting_bins[:-1])),
        starting_hist,
        label="Initial",
    )

    # Set up labelling conventions
    # externals = [2, 9, 12, 19]
    if args.all:
        externals = [i for i in range(20)]
        label_dict = {i: f"O{i}_up" if i < 10 else f"O{i}_down" for i in range(20)}
    else:
        externals = [int(output) for output in rg_config.outputs]
        label_dict = {
            2: "O3_up",
            9: "O10_up",
            10: "O1_down",
            17: "O8_down",
        }

    all_data = defaultdict()
    # Run the solver and extract each output
    data_sum = np.full(shape=(n, 1), fill_value=0.0, dtype=np.float64)
    over_mask_phis = {}
    for index in externals:
        data = numerical_solver(
            starting_t,
            starting_f,
            starting_phases,
            n,
            index,
            inputs,
            rg_config.matrix_batch_size,
        )
        all_data[f"{index}"] = data
        get_memory_usage(f"Memory usage after solving for output {index}")
        print("-" * 100)

        print(f"Printing stats and plotting data for output {index}")
        # Get some statistics
        over_mask = data > 1.0
        under_mask = data < 0.0

        # # Check the values of phi where data is outside the expected domain
        # over_mask_phi_vals = starting_phases[over_mask, :]
        # over_mask_phis.update({index: over_mask_phi_vals})
        # Print the relevant stats
        print(f"Min t = {np.min(data)}, Max t = {np.max(data)}")
        print(
            f"t values > 1.0 = {over_mask.sum()}, t values < 0.0 = {under_mask.sum()}"
        )
        print(f"Mean = {np.mean(data)}, Median = {np.median(data)}")
        if np.max(data) - 1.0 <= 1e-12 and np.min(data) >= -1e-12:
            data = np.clip(data, 0.0, 1.0)
        # Plot the data
        hist, bins = np.histogram(data, rg_config.t_bins, rg_config.t_range)
        density = get_density(hist, bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(centers, density, label=f"{label_dict[index]}")
        data_sum = data_sum + np.abs(data) ** 2
        print("-" * 100)

    get_memory_usage("Memory usage after handling all 4 outputs")
    print("-" * 100)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    output_plot_file_name = f"{plots_folder}/qshe_{n}_outputs_f_{args.f}.png"
    plt.savefig(output_plot_file_name, dpi=150)
    plt.close("outputs")
    print(f"Outputs plot saved to {output_plot_file_name}")

    get_memory_usage("Overall memory usage for this analysis")
    print(
        f"Overall analysis done on {get_current_date()} after {time() - start_time:.3f} seconds"
    )
    print("-" * 100)
    print("Computing stats for |O3_up|^2 + |O10_up|^2 + |O1_down|^2 + |O8_down|^2:")
    print(f"Mean of output sum: {np.mean(data_sum)}")
    print(f"Median of output sum: {np.median(data_sum)}")
    print(f"Min of output sum: {np.min(data_sum)}")
    print(f"Max of output sum: {np.max(data_sum)}")
    max_output_err = np.max(np.abs(data_sum - 1.0))
    if max_output_err > 1e-12:
        print(f"The difference exceeds given tolerance 1e-12 : {max_output_err}")
    else:
        print(f"|Max output error - 1.0| = {max_output_err}")
    print("=" * 100)
    output_file.close()
    sys.stdout = original_output
    # print(over_mask_phis.keys())
    print(f"Overall analysis done after {time() - start_time:.3f} seconds")
    print(f"Program completed on {get_current_date()}")

    t_prime = all_data["2"]
    f_prime = all_data["17"]
    z = convert_t_to_z(t_prime)

    # hists = build_2d_hist(
    #     z,
    #     f_prime,
    #     rg_config.z_bins,
    #     rg_config.t_bins,
    #     rg_config.z_range,
    #     rg_config.t_range,
    #     True,
    # )
    zbins = 200
    fbins = 100
    hists = build_2d_hist(
        z,
        f_prime,
        zbins,
        fbins,
        rg_config.z_range,
        rg_config.t_range,
        True,
    )
    plot_2d_hist(hists)

    z_sample, f_sample = rejection_sampler_2d(hists, rng, n)
    # z_sample, f_sample = inverse_cdf_2d(hists, rng, n)
    t_sample = convert_z_to_t(z_sample)
    z_hist, z_bins = np.histogram(z_sample, bins=hists["z"]["binedges"])
    f_hist, f_bins = np.histogram(f_sample, bins=hists["f"]["binedges"])

    # z_unsym_hist, z_unsym_bins = np.histogram(z, bins=200)
    # f_unsym_hist, f_unsym_bins = np.histogram(f_prime, bins=100)

    z_con, f_con = conditional_2d_resampler(hists, rng, n)
    t_con = convert_z_to_t(z_con)
    zcon_hist, zcon_bins = np.histogram(z_con, bins=hists["z"]["binedges"])
    fcon_hist, fcon_bins = np.histogram(f_con, bins=hists["f"]["binedges"])

    print(np.max(np.abs(f_sample)[:, None] ** 2 + np.abs(t_sample)[:, None] ** 2))
    print(np.max(np.abs(f_prime)[:, None] ** 2 + np.abs(t_prime)[:, None] ** 2))
    print(np.max(np.abs(f_con)[:, None] ** 2 + np.abs(t_con)[:, None] ** 2))

    z_com_mean, z_com_std = hist_moments(hists["z"]["counts"], hists["z"]["binedges"])
    z_rej_mean, z_rej_std = hist_moments(z_hist, z_bins)
    z_con_mean, z_con_std = hist_moments(zcon_hist, zcon_bins)
    f_com_mean, f_com_std = hist_moments(hists["f"]["counts"], hists["f"]["binedges"])
    f_rej_mean, f_rej_std = hist_moments(f_hist, f_bins)
    f_con_mean, f_con_std = hist_moments(fcon_hist, fcon_bins)

    print(
        f"Computed: Z Mean = {z_com_mean}, Z STD = {z_com_std}, F Mean = {f_com_mean}, f STD = {f_com_std}"
    )
    print(
        f"Rejection: Z Mean = {z_rej_mean}, Z STD = {z_rej_std}, F Mean = {f_rej_mean}, f STD = {f_rej_std}"
    )
    print(
        f"Conditional: Z Mean = {z_con_mean}, Z STD = {z_con_std}, F Mean = {f_con_mean}, f STD = {f_con_std}"
    )
    # print(np.min(z_unsym_hist), np.max(z_unsym_hist))
    # print(np.min(z), np.max(z))
    # print(np.median(z_hist), np.median(hists["z"]["counts"]))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))
    ax0.plot(hists["z"]["bincenters"], hists["z"]["counts"], label="z computed")
    ax0.plot(0.5 * (z_bins[1:] + z_bins[:-1]), z_hist, label="z rej sample")
    ax0.plot(0.5 * (zcon_bins[1:] + zcon_bins[:-1]), zcon_hist, label="z con sample")
    # ax0.plot(0.5 * (z_unsym_bins[1:] + z_unsym_bins[:-1]), z_unsym_hist, label="z 1d")
    ax1.plot(hists["f"]["bincenters"], hists["f"]["counts"], label="f computed")
    ax1.plot(0.5 * (f_bins[1:] + f_bins[:-1]), f_hist, label="f rej sample")
    ax1.plot(0.5 * (fcon_bins[1:] + fcon_bins[:-1]), fcon_hist, label="f con sample")
    # ax1.plot(0.5 * (f_unsym_bins[1:] + f_unsym_bins[:-1]), f_unsym_hist, label="f 1d")
    ax0.set_title("Distributions of z for different sampling methods")
    ax1.set_title("Distributions of f for different sampling methods")
    ax0.legend()
    ax1.legend()
    plt.show()
