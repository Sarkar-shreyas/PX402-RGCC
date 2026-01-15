# flake8: noqa: E501
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from source.utilities import (
    generate_constant_array,
    generate_initial_t_distribution,
    generate_random_phases,
    extract_t_samples,
    get_current_date,
    get_density,
    build_rng,
    launder,
)
from source.config import build_config, handle_config, RGConfig
from source.parse_config import build_parser
from time import time
import psutil
import os
import sys
from constants import T_DICT, qshe_dir, data_dir, PHI_DICT

process = psutil.Process(os.getpid())


def generate_phase_array(n: int, dimensions: int, phi: float) -> np.ndarray:
    """Populates the phase arrays with a constant phi value"""
    return np.full(shape=(n, dimensions), fill_value=phi, dtype=np.float64)


def get_memory_usage(item: str = "") -> None:
    """Prints the memory usage for a process"""
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"{item} = {memory:.3f} MB")


def solve_qshe_matrix_eq(
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    split: float,
    batch_size: int,
    output_index: int,
):
    """Build the 20x20 matrix equation and solve Mx = b"""
    t1, t2, t3, t4, t5 = ts.T
    r1 = np.sqrt(split - t1 * t1)
    r2 = np.sqrt(split - t2 * t2)
    r3 = np.sqrt(split - t3 * t3)
    r4 = np.sqrt(split - t4 * t4)
    r5 = np.sqrt(split - t5 * t5)
    f1, f2, f3, f4, f5 = fs.T

    (
        phi12,
        phi13,
        phi15,
        phi21,
        phi23,
        phi24,
        phi31,
        phi32,
        phi34,
        phi35,
        phi42,
        phi43,
        phi45,
        phi51,
        phi53,
        phi54,
    ) = phis.T

    # Define our matrices
    M = np.zeros((batch_size, 20, 20), dtype=np.complex128)
    b = np.zeros((batch_size, 20, 1), dtype=np.complex128)

    # Now we need to assign data for 20 [0-19] rows... TODO: See if there's a more efficient way at some point
    # Matrix M
    # Row 0
    M[:, 0, 0] = 1
    M[:, 0, 4] = -r1 * np.exp(1j * phi31)
    M[:, 0, 18] = -f1 * np.exp(1j * phi51)

    # Row 1
    M[:, 1, 1] = 1
    M[:, 1, 4] = t1 * np.exp(1j * phi31)
    M[:, 1, 12] = f1 * np.exp(1j * phi21)

    # Row 2
    M[:, 2, 0] = -t2 * np.exp(1j * phi12)
    M[:, 2, 2] = 1
    M[:, 2, 6] = -r2 * np.exp(1j * phi42)
    M[:, 2, 15] = -f2 * np.exp(1j * phi32)

    # Row 3
    M[:, 3, 0] = -r2 * np.exp(1j * phi12)
    M[:, 3, 3] = 1
    M[:, 3, 6] = t2 * np.exp(1j * phi42)

    # Row 4
    M[:, 4, 3] = -r3 * np.exp(1j * phi23)
    M[:, 4, 4] = 1
    M[:, 4, 8] = -t3 * np.exp(1j * phi53)
    M[:, 4, 16] = -f3 * np.exp(1j * phi43)

    # Row 5
    M[:, 5, 3] = t3 * np.exp(1j * phi23)
    M[:, 5, 5] = 1
    M[:, 5, 8] = -r3 * np.exp(1j * phi53)
    M[:, 5, 11] = f3 * np.exp(1j * phi13)

    # Row 6
    M[:, 6, 5] = -t4 * np.exp(1j * phi34)
    M[:, 6, 6] = 1
    M[:, 6, 19] = -f4 * np.exp(1j * phi54)

    # Row 7
    M[:, 7, 5] = -r4 * np.exp(1j * phi34)
    M[:, 7, 7] = 1
    M[:, 7, 13] = f4 * np.exp(1j * phi24)

    # Row 8
    M[:, 8, 1] = -t5 * np.exp(1j * phi15)
    M[:, 8, 7] = -r5 * np.exp(1j * phi45)
    M[:, 8, 8] = 1

    # Row 9
    M[:, 9, 1] = -r5 * np.exp(1j * phi15)
    M[:, 9, 7] = t5 * np.exp(1j * phi45)
    M[:, 9, 9] = 1
    M[:, 9, 14] = f5 * np.exp(1j * phi35)

    # Row 10
    M[:, 10, 4] = f1 * np.exp(1j * phi31)
    M[:, 10, 10] = 1
    M[:, 10, 12] = -t1 * np.exp(1j * phi21)
    M[:, 10, 18] = -r1 * np.exp(1j * phi51)

    # Row 11
    M[:, 11, 11] = 1
    M[:, 11, 12] = -r1 * np.exp(1j * phi21)
    M[:, 11, 18] = t1 * np.exp(1j * phi51)

    # Row 12
    M[:, 12, 6] = f2 * np.exp(1j * phi42)
    M[:, 12, 12] = 1
    M[:, 12, 15] = -r2 * np.exp(1j * phi32)

    # Row 13
    M[:, 13, 0] = -f2 * np.exp(1j * phi12)
    M[:, 13, 13] = 1
    M[:, 13, 15] = t2 * np.exp(1j * phi32)

    # Row 14
    M[:, 14, 3] = f3 * np.exp(1j * phi23)
    M[:, 14, 11] = -t3 * np.exp(1j * phi13)
    M[:, 14, 14] = 1
    M[:, 14, 16] = -r3 * np.exp(1j * phi43)

    # Row 15
    M[:, 15, 8] = -f3 * np.exp(1j * phi53)
    M[:, 15, 11] = -r3 * np.exp(1j * phi13)
    M[:, 15, 15] = 1
    M[:, 15, 16] = t3 * np.exp(1j * phi43)

    # Row 16
    M[:, 16, 13] = -t4 * np.exp(1j * phi24)
    M[:, 16, 16] = 1
    M[:, 16, 19] = -r4 * np.exp(1j * phi54)

    # Row 17
    M[:, 17, 5] = -f4 * np.exp(1j * phi34)
    M[:, 17, 13] = -r4 * np.exp(1j * phi24)
    M[:, 17, 17] = 1
    M[:, 17, 19] = t4 * np.exp(1j * phi54)

    # Row 18
    M[:, 18, 7] = f5 * np.exp(1j * phi45)
    M[:, 18, 14] = -t5 * np.exp(1j * phi35)
    M[:, 18, 18] = 1

    # Row 19
    M[:, 19, 1] = -f5 * np.exp(1j * phi15)
    M[:, 19, 14] = -r5 * np.exp(1j * phi35)
    M[:, 19, 19] = 1
    # Set values for the 4 Inputs for testing
    I1_up = 1.0
    I3_down = 0.0
    I10_down = 0.0
    I8_up = 0.0
    # # b matrix for M
    # b[:, 0, 0] = r1 * tau1 * I1
    # b[:, 1, 0] = 1j * t1 * tau1 * I1
    # b[:, 2, 0] = -f1 * I1
    # b[:, 5, 0] = f2 * I2
    # b[:, 6, 0] = 1j * t2 * tau2 * I2
    # b[:, 7, 0] = r2 * tau2 * I2
    # b[:, 12, 0] = 1j * t4 * tau4 * I3
    # b[:, 13, 0] = r4 * tau4 * I3
    # b[:, 15, 0] = f4 * I3
    # b[:, 16, 0] = f5 * I4
    # b[:, 18, 0] = r5 * tau5 * I4
    # b[:, 19, 0] = 1j * t5 * tau5 * I4

    # b matrix for M2
    b[:, 0, 0] = t1 * I1_up
    b[:, 1, 0] = r1 * I1_up
    b[:, 3, 0] = -f2 * I3_down
    b[:, 6, 0] = r4 * I8_up
    b[:, 7, 0] = -t4 * I8_up
    b[:, 8, 0] = f5 * I10_down
    b[:, 11, 0] = f1 * I1_up
    b[:, 12, 0] = t2 * I3_down
    b[:, 13, 0] = r2 * I3_down
    b[:, 16, 0] = -f4 * I8_up
    b[:, 18, 0] = r5 * I10_down
    b[:, 19, 0] = -t5 * I10_down

    x = np.linalg.solve(M, b)

    # Outputs are index 2, 9, 10 and 17 in order of O3_up, O10_up, O1_down and O8_down
    # return x
    return x[:, output_index]


def numerical_solver(
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    split: float,
    N: int,
    output_index: int,
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
            solve_qshe_matrix_eq(
                ts[indexes], fs[indexes], phis[indexes], split, batch_size, output_index
            )
        )
        if (i + 1) in {10, 50, 100, num_batches}:
            # get_memory_usage(f"Memory usage after batch {i + 1}")
            print(f"Batch {i + 1} done in {time() - start:.3f} seconds")
    print("-" * 100)
    print(
        f"Computation for all {num_batches} batches of index {output_index} done after {time() - start:.3f} seconds"
    )
    return np.abs(output)


def check_single_node():
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


def gen_initial_data(
    config: RGConfig,
    t_val: int,
    phi_val: int,
    f_val: float,
    rng: np.random.Generator,
    fp_data,
) -> dict:
    n = config.samples
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
        split = 1 - f_val**2
        t_array = rng.uniform(0, np.sqrt(split), size=(n, 5))
    else:
        t_array = generate_constant_array(n, T_DICT[str(t_val)], 5)
    if phi_val == 0:
        phi_array = generate_random_phases(n, rng, 16)
    else:
        phi_array = generate_constant_array(n, PHI_DICT[str(phi_val)], 16)
    data_dict = {"t": t_array, "f": f_array, "phi": phi_array, "split": split}
    return data_dict


if __name__ == "__main__":
    print(f"Program started on {get_current_date()}")
    start_time = time()
    parser = build_parser()
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
    args = parser.parse_args()
    config = handle_config(args.config, args.override)
    rg_config = build_config(config)
    n = rg_config.samples
    num_phases = 16
    rng = build_rng(rg_config.seed)
    check_single_node()
    print(f"t = {args.t}, phi = {args.phi}, f = {args.f}")
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
    initial_data = gen_initial_data(rg_config, args.t, args.phi, args.f, rng, fp_data)
    starting_t = initial_data["t"]
    starting_f = initial_data["f"]
    starting_phases = initial_data["phi"]
    split = initial_data["split"]
    # # starting_t = launder(n, fp_density, fp_bins, fp_centers, "i")
    # starting_t = generate_initial_t_distribution(n, rng)
    # # starting_t = np.full(shape=n, fill_value=1 / np.sqrt(2), dtype=np.float64)
    # # starting_tau = generate_initial_t_distribution(n)
    # starting_tau = np.full(shape=n, fill_value=0.9, dtype=np.float64)
    # # starting_tau = np.random.uniform(0, 1, n)
    # phases = generate_random_phases(n, rng, num_phases)
    # # phases = generate_phase_array(n, num_phases, np.pi / 2)
    # t_samples = extract_t_samples(starting_t, n, rng)
    # tau_samples = extract_t_samples(starting_tau, n, rng)
    print(
        f"Initial t, f and phi arrays generated after {time() - start_time:.3f} seconds"
    )
    # get_memory_usage("Memory usage after generating initial data")
    print("-" * 100)
    # Set up initial plots
    plt.figure(num="outputs", figsize=(12, 6))
    plt.title("Distribution of outputs")
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
            split,
            n,
            index,
            rg_config.matrix_batch_size,
        )
        all_data[f"{index}"] = data
        # clipped_data = np.clip(data, 0.0, 1.0)
        # r_data = np.sqrt(1 - data * data)
        # clipped_r = np.clip(r_data, 0.0, 1.0)
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
        # plt.scatter(data[0], data[0], label=f"{label_dict[index]}")
        print("-" * 100)
    get_memory_usage("Memory usage after handling all 4 outputs")
    print("-" * 100)
    # o3_up = all_data["2"]
    # o10_up = all_data["9"]
    # o1_down = all_data["12"]
    # o8_down = all_data["19"]
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    output_plot_file_name = f"{plots_folder}/qshe_{n}_outputs_f_{args.f}.png"
    plt.savefig(output_plot_file_name, dpi=150)
    plt.close("outputs")
    print(f"Outputs plot saved to {output_plot_file_name}")

    # Now extract individual renormalised parameters and plot them
    # f_prime = o8_down
    # tau_prime = np.sqrt(1 - f_prime**2)
    # t_prime = o3_up / tau_prime
    # r_prime = o10_up / tau_prime

    # renormalised_data = [f_prime, tau_prime, t_prime, r_prime]
    # # Analyse the data obtained
    # print("=" * 100)
    # for _ in renormalised_data:
    #     over = _ > 1.0
    #     under = _ < 0.0
    #     print(f"Min = {np.min(_)}, Max = {np.max(_)}")
    #     print(f"Values above 1.0 = {over.sum()}, Values under 0.0 = {under.sum()}")
    #     print(f"Mean = {np.mean(_)}, Median = {np.median(_)}")
    #     print("-" * 100)
    # # Set up the figure for the renormalised variables
    # plt.figure("variables", figsize=(12, 6))
    # plt.title("Distribution of renormalised parameters")
    # plt.xlabel("var")
    # plt.ylabel("P(var)")
    # plt.xlim((0, 1))
    # plt.ylim((0, 4))

    # # Build histograms for the 4 renormalised parameters
    # f_prime_hist, f_prime_bins = np.histogram(f_prime, rg_config.t_bins, rg_config.t_range, density=True)
    # f_prime_centers = 0.5 * (f_prime_bins[1:] + f_prime_bins[:-1])
    # tau_prime_hist, tau_prime_bins = np.histogram(
    #     tau_prime, rg_config.t_bins, rg_config.t_range, density=True
    # )
    # tau_prime_centers = 0.5 * (tau_prime_bins[1:] + tau_prime_bins[:-1])
    # t_prime_hist, t_prime_bins = np.histogram(t_prime, rg_config.t_bins, rg_config.t_range, density=True)
    # t_prime_centers = 0.5 * (t_prime_bins[1:] + t_prime_bins[:-1])
    # r_prime_hist, r_prime_bins = np.histogram(r_prime, rg_config.t_bins, rg_config.t_range, density=True)
    # r_prime_centers = 0.5 * (r_prime_bins[1:] + r_prime_bins[:-1])

    # # Plot the renormalised parameters
    # plt.plot(f_prime_centers, f_prime_hist, label="f'")
    # plt.plot(tau_prime_centers, tau_prime_hist, label="tau'")
    # plt.plot(t_prime_centers, t_prime_hist, label="t'")
    # plt.plot(r_prime_centers, r_prime_hist, label="r'")
    # plt.legend(loc="upper left")
    # var_plot_file_name = f"{plots_folder}/qshe_vars_dist_{n}.png"
    # plt.savefig(var_plot_file_name, dpi=150)
    # plt.close("variables")
    # print(f"Variables plot saved to {var_plot_file_name}")

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
    print("=" * 100)
    output_file.close()
    sys.stdout = original_output
    # print(over_mask_phis.keys())
    print(
        f"Overall analysis done on {get_current_date()} after {time() - start_time:.3f} seconds"
    )
