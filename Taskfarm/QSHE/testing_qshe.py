# flake8: noqa: E501
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from source.utilities import (
    T_RANGE,
    T_BINS,
    generate_initial_t_distribution,
    generate_random_phases,
    extract_t_samples,
    get_current_date,
)
from time import time
import psutil
import os
import sys
from constants import taskfarm_dir

process = psutil.Process(os.getpid())


def get_memory_usage(item: str = "") -> None:
    """Prints the memory usage for a process"""
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"{item} = {memory:.3f} MB")


def solve_qshe_matrix_eq(
    ts: np.ndarray,
    taus: np.ndarray,
    phis: np.ndarray,
    batch_size: int,
    output_index: int,
):
    """Build the 20x20 matrix equation and solve Mx = b"""
    t1, t2, t3, t4, t5 = ts.T
    tau1, tau2, tau3, tau4, tau5 = taus.T
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)
    f1 = np.sqrt(1 - tau1 * tau1)
    f2 = np.sqrt(1 - tau2 * tau2)
    f3 = np.sqrt(1 - tau3 * tau3)
    f4 = np.sqrt(1 - tau4 * tau4)
    f5 = np.sqrt(1 - tau5 * tau5)

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
    # # Row 0
    # M[:, 0, 0] = 1
    # M[:, 0, 9] = t1 * tau1 * np.exp(1j * phi31)
    # M[:, 0, 19] = -f1 * np.exp(1j * phi51)

    # # Row 1
    # M[:, 1, 1] = 1
    # M[:, 1, 7] = f1 * np.exp(1j * phi21)
    # M[:, 1, 9] = -r1 * tau1 * np.exp(1j * phi31)

    # # Row 2
    # M[:, 2, 2] = 1
    # M[:, 2, 7] = t1 * tau1 * np.exp(1j * phi21)
    # M[:, 2, 19] = -r1 * tau1 * np.exp(1j * phi51)

    # # Row 3
    # M[:, 3, 3] = 1
    # M[:, 3, 7] = -r1 * tau1 * np.exp(1j * phi21)
    # M[:, 3, 9] = -f1 * np.exp(1j * phi31)
    # M[:, 3, 19] = t1 * tau1 * np.exp(1j * phi51)

    # # Row 4
    # M[:, 4, 0] = -r2 * tau2 * np.exp(1j * phi12)
    # M[:, 4, 4] = 1
    # M[:, 4, 11] = -f2 * np.exp(1j * phi32)
    # M[:, 4, 12] = t2 * tau2 * np.exp(1j * phi42)

    # # Row 5
    # M[:, 5, 0] = t2 * tau2 * np.exp(1j * phi12)
    # M[:, 5, 5] = 1
    # M[:, 5, 12] = -r2 * tau2 * np.exp(1j * phi42)

    # # Row 6
    # M[:, 6, 0] = f2 * np.exp(1j * phi12)
    # M[:, 6, 6] = 1
    # M[:, 6, 11] = -r2 * tau2 * np.exp(1j * phi32)

    # # Row 7
    # M[:, 7, 7] = 1
    # M[:, 7, 11] = t2 * tau2 * np.exp(1j * phi32)
    # M[:, 7, 12] = -f2 * np.exp(1j * phi42)

    # # Row 8
    # M[:, 8, 2] = -f3 * np.exp(1j * phi13)
    # M[:, 8, 5] = -r3 * tau3 * np.exp(1j * phi23)
    # M[:, 8, 8] = 1
    # M[:, 8, 16] = t3 * tau3 * np.exp(1j * phi53)

    # # Row 9
    # M[:, 9, 5] = t3 * tau3 * np.exp(1j * phi23)
    # M[:, 9, 9] = 1
    # M[:, 9, 15] = f3 * np.exp(1j * phi43)
    # M[:, 9, 16] = -r3 * tau3 * np.exp(1j * phi53)

    # # Row 10
    # M[:, 10, 2] = -r3 * tau3 * np.exp(1j * phi13)
    # M[:, 10, 5] = f3 * np.exp(1j * phi23)
    # M[:, 10, 10] = 1
    # M[:, 10, 15] = t3 * tau3 * np.exp(1j * phi43)

    # # Row 11
    # M[:, 11, 2] = t3 * tau3 * np.exp(1j * phi13)
    # M[:, 11, 11] = 1
    # M[:, 11, 15] = -r3 * tau3 * np.exp(1j * phi43)
    # M[:, 11, 16] = -f3 * np.exp(1j * phi53)

    # # Row 12
    # M[:, 12, 8] = -r4 * tau4 * np.exp(1j * phi34)
    # M[:, 12, 12] = 1
    # M[:, 12, 17] = -f4 * np.exp(1j * phi54)

    # # Row 13
    # M[:, 13, 6] = f4 * np.exp(1j * phi24)
    # M[:, 13, 8] = t4 * tau4 * np.exp(1j * phi34)
    # M[:, 13, 13] = 1

    # # Row 14
    # M[:, 14, 6] = t4 * tau4 * np.exp(1j * phi24)
    # M[:, 14, 8] = f4 * np.exp(1j * phi34)
    # M[:, 14, 14] = 1
    # M[:, 14, 18] = -r4 * tau4 * np.exp(1j * phi54)

    # # Row 15
    # M[:, 15, 6] = -r4 * tau4 * np.exp(1j * phi24)
    # M[:, 15, 15] = 1
    # M[:, 15, 18] = t4 * tau4 * np.exp(1j * phi54)

    # # Row 16
    # M[:, 16, 1] = -r5 * tau5 * np.exp(1j * phi15)
    # M[:, 16, 13] = t5 * tau5 * np.exp(1j * phi45)
    # M[:, 16, 16] = 1

    # # Row 17
    # M[:, 17, 1] = t5 * tau5 * np.exp(1j * phi15)
    # M[:, 17, 10] = f5 * np.exp(1j * phi35)
    # M[:, 17, 13] = -r5 * tau5 * np.exp(1j * phi45)
    # M[:, 17, 17] = 1

    # # Row 18
    # M[:, 18, 1] = f5 * np.exp(1j * phi15)
    # M[:, 18, 10] = t5 * tau5 * np.exp(1j * phi35)
    # M[:, 18, 18] = 1

    # # Row 19
    # M[:, 19, 10] = -r5 * tau5 * np.exp(1j * phi35)
    # M[:, 19, 13] = -f5 * np.exp(1j * phi45)
    # M[:, 19, 19] = 1

    # Matrix M2
    # Row 0
    M[:, 0, 0] = 1
    M[:, 0, 4] = -r1 * tau1 * np.exp(1j * phi31)
    M[:, 0, 18] = -f1 * np.exp(1j * phi51)

    # Row 1
    M[:, 1, 1] = 1
    M[:, 1, 4] = t1 * tau1 * np.exp(1j * phi31)
    M[:, 1, 12] = f1 * np.exp(1j * phi21)

    # Row 2
    M[:, 2, 0] = -t2 * tau2 * np.exp(1j * phi12)
    M[:, 2, 2] = 1
    M[:, 2, 6] = -r2 * tau2 * np.exp(1j * phi42)
    M[:, 2, 15] = -f2 * np.exp(1j * phi32)

    # Row 3
    M[:, 3, 0] = -r2 * tau2 * np.exp(1j * phi12)
    M[:, 3, 3] = 1
    M[:, 3, 6] = t2 * tau2 * np.exp(1j * phi42)

    # Row 4
    M[:, 4, 3] = -r3 * tau3 * np.exp(1j * phi23)
    M[:, 4, 4] = 1
    M[:, 4, 8] = -t3 * tau3 * np.exp(1j * phi53)
    M[:, 4, 16] = -f3 * np.exp(1j * phi43)

    # Row 5
    M[:, 5, 3] = t3 * tau3 * np.exp(1j * phi23)
    M[:, 5, 5] = 1
    M[:, 5, 8] = -r3 * tau3 * np.exp(1j * phi53)
    M[:, 5, 11] = f3 * np.exp(1j * phi13)

    # Row 6
    M[:, 6, 5] = -t4 * tau4 * np.exp(1j * phi34)
    M[:, 6, 6] = 1
    M[:, 6, 19] = -f4 * np.exp(1j * phi54)

    # Row 7
    M[:, 7, 5] = -r4 * tau4 * np.exp(1j * phi34)
    M[:, 7, 7] = 1
    M[:, 7, 13] = f4 * np.exp(1j * phi24)

    # Row 8
    M[:, 8, 1] = -t5 * tau5 * np.exp(1j * phi15)
    M[:, 8, 7] = -r5 * tau5 * np.exp(1j * phi45)
    M[:, 8, 8] = 1

    # Row 9
    M[:, 9, 1] = -r5 * tau5 * np.exp(1j * phi15)
    M[:, 9, 7] = t5 * tau5 * np.exp(1j * phi45)
    M[:, 9, 9] = 1
    M[:, 9, 14] = f5 * np.exp(1j * phi35)

    # Row 10
    M[:, 10, 10] = 1
    M[:, 10, 12] = -t1 * tau1 * np.exp(1j * phi21)
    M[:, 10, 14] = f1 * np.exp(1j * phi31)
    M[:, 10, 18] = -r1 * tau1 * np.exp(1j * phi51)

    # Row 11
    M[:, 11, 11] = 1
    M[:, 11, 12] = -r1 * tau1 * np.exp(1j * phi21)
    M[:, 11, 18] = t1 * tau1 * np.exp(1j * phi51)

    # Row 12
    M[:, 12, 6] = f2 * np.exp(1j * phi42)
    M[:, 12, 12] = 1
    M[:, 12, 15] = -r2 * tau2 * np.exp(1j * phi32)

    # Row 13
    M[:, 13, 0] = -f2 * np.exp(1j * phi12)
    M[:, 13, 13] = 1
    M[:, 13, 15] = t2 * tau2 * np.exp(1j * phi32)

    # Row 14
    M[:, 14, 3] = f3 * np.exp(1j * phi23)
    M[:, 14, 11] = -t3 * tau3 * np.exp(1j * phi13)
    M[:, 14, 16] = -r3 * tau3 * np.exp(1j * phi43)
    M[:, 14, 14] = 1

    # Row 15
    M[:, 15, 8] = -f3 * np.exp(1j * phi53)
    M[:, 15, 11] = -r3 * tau3 * np.exp(1j * phi13)
    M[:, 15, 15] = 1
    M[:, 15, 16] = t3 * tau3 * np.exp(1j * phi43)

    # Row 16
    M[:, 16, 13] = -t4 * tau4 * np.exp(1j * phi24)
    M[:, 16, 16] = 1
    M[:, 16, 19] = -r4 * tau4 * np.exp(1j * phi54)

    # Row 17
    M[:, 17, 5] = -f4 * np.exp(1j * phi34)
    M[:, 17, 13] = -r4 * tau4 * np.exp(1j * phi24)
    M[:, 17, 17] = 1
    M[:, 17, 19] = t4 * tau4 * np.exp(1j * phi54)

    # Row 18
    M[:, 18, 7] = f5 * np.exp(1j * phi45)
    M[:, 18, 14] = -t5 * tau5 * np.exp(1j * phi35)
    M[:, 18, 18] = 1

    # Row 19
    M[:, 19, 1] = -f5 * np.exp(1j * phi15)
    M[:, 19, 14] = -r5 * tau5 * np.exp(1j * phi35)
    M[:, 19, 19] = 1
    # Set values for the 4 Inputs for testing
    I1_up = 1.0
    I3_down = 0.0
    I8_up = 0.0
    I10_down = 0.0
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
    b[:, 0, 0] = t1 * tau1 * I1_up
    b[:, 1, 0] = r1 * tau1 * I1_up
    b[:, 3, 0] = -f2 * I3_down
    b[:, 6, 0] = r4 * tau4 * I8_up
    b[:, 7, 0] = -t4 * tau4 * I8_up
    b[:, 8, 0] = f5 * I10_down
    b[:, 11, 0] = f1 * I1_up
    b[:, 12, 0] = t2 * tau2 * I3_down
    b[:, 13, 0] = r2 * tau2 * I3_down
    b[:, 16, 0] = -f4 * I8_up
    b[:, 18, 0] = r5 * tau5 * I10_down
    b[:, 19, 0] = -t5 * tau5 * I10_down

    x = np.linalg.solve(M, b)

    # Outputs are index 2, 9, 10 and 17 in order of O3_up, O10_up, O1_down and O8_down
    # return x
    return x[:, output_index]


def numerical_solver(
    ts: np.ndarray, taus: np.ndarray, phis: np.ndarray, N: int, output_index: int
) -> np.ndarray:
    """Solve the matrix equation Mx=b for N samples using batching"""
    start = time()
    batch_size = 100000
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
                ts[indexes], taus[indexes], phis[indexes], batch_size, output_index
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
    t = np.random.uniform(0, 1, 1000)
    tau = np.random.uniform(0, 1, 1000)
    r = np.sqrt(1 - t**2)
    f = np.sqrt(1 - tau**2)

    i1 = np.full(shape=(1000, 1), fill_value=1.0)
    i2 = np.full(shape=(1000, 1), fill_value=0.0)
    i3 = np.full(shape=(1000, 1), fill_value=0.0)
    i4 = np.full(shape=(1000, 1), fill_value=0.0)
    I = np.concatenate([i1, i2, i3, i4], axis=1)
    S = np.zeros(shape=(1000, 4, 4), dtype=np.complex128)
    S[:, 0, 0] = t * tau
    S[:, 0, 1] = r * tau
    S[:, 0, 3] = f
    S[:, 1, 1] = -t * tau
    S[:, 1, 0] = r * tau
    S[:, 1, 2] = -f
    S[:, 2, 2] = t * tau
    S[:, 2, 3] = r * tau
    S[:, 2, 1] = -f
    S[:, 3, 3] = -t * tau
    S[:, 3, 2] = r * tau
    S[:, 3, 0] = f

    x = np.linalg.solve(S, I)
    print(np.abs(x))


if __name__ == "__main__":
    print(f"Program started on {get_current_date()}")
    start_time = time()
    n = int(1e6)
    num_phases = 16
    original_output = sys.stdout
    output_folder = f"{taskfarm_dir}/QSHE/outputs"
    plots_folder = f"{taskfarm_dir}/QSHE/plots"
    day_output_folder = f"{output_folder}/{get_current_date('day')}"
    hour_output_folder = f"{output_folder}/{get_current_date('hour')}"
    output_file_name = f"{hour_output_folder}/qshe_{n}_outputs.txt"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(day_output_folder, exist_ok=True)
    os.makedirs(hour_output_folder, exist_ok=True)
    print(f"Output being printed to {output_file_name}")
    output_file = open(output_file_name, "w")
    sys.stdout = output_file
    print(f"Program started on {get_current_date()}")
    print(f"Starting numerical solver for QSHE matrix with {n} samples")
    print("-" * 100)
    # Generate initial arrays
    # check_single_node()
    # get_memory_usage("Memory usage before generating initial data")
    starting_t = generate_initial_t_distribution(n)
    # starting_t = np.random.uniform(0, 1, n)
    # starting_tau = generate_initial_t_distribution(n)
    starting_tau = np.full(shape=(n), fill_value=0.5, dtype=np.float64)
    # starting_tau = np.random.uniform(0, 1, n)
    phases = generate_random_phases(n, num_phases)
    t_samples = extract_t_samples(starting_t, n)
    tau_samples = extract_t_samples(starting_tau, n)
    print(
        f"Initial t, tau and phi arrays generated after {time() - start_time:.3f} seconds"
    )
    # get_memory_usage("Memory usage after generating initial data")
    print("-" * 100)
    # Run the solver
    plt.figure(figsize=(12, 6))
    plt.title("Distribution of t")
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.ylim((0, 4))
    starting_hist, starting_bins = np.histogram(
        starting_t, T_BINS, T_RANGE, density=True
    )
    plt.plot(
        (0.5 * (starting_bins[1:] + starting_bins[:-1])),
        starting_hist,
        label="Initial",
    )
    externals = [2, 9, 10, 17]
    all_data = defaultdict()
    for index in externals:
        data = numerical_solver(t_samples, tau_samples, phases, n, index)
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
        print(f"Min t = {np.min(data):.5f}, Max t = {np.max(data):.5f}")
        print(
            f"t values > 1.0 = {over_mask.sum()}, t values < 0.0 = {under_mask.sum()}"
        )
        print(f"Mean = {np.mean(data):.5f}, Median = {np.median(data):.5f}")
        # Plot the data
        hist, bins = np.histogram(data, T_BINS, T_RANGE, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=f"Raw_t_index_{index}")
        # clipped_hist, clipped_bins = np.histogram(
        #     clipped_data, T_BINS, T_RANGE, density=True
        # )
        # r_hist, r_bins = np.histogram(r_data, T_BINS, T_RANGE, density=True)
        # clipped_r_hist, clipped_r_bins = np.histogram(
        #     clipped_r, T_BINS, T_RANGE, density=True
        # )
        # plt.plot(centers, clipped_hist, label="Clipped t data")
        # plt.plot(centers, r_hist, label="Raw r data")
        # plt.plot(centers, clipped_r_hist, label="Clipped r data")
        print("-" * 100)
    get_memory_usage("Memory usage after handling all 4 outputs")
    print("-" * 100)
    o3_up = all_data["2"]
    o10_up = all_data["9"]
    o1_down = all_data["10"]
    o8_down = all_data["17"]
    k = 0
    for i in [o3_up, o10_up, o8_down]:
        test_data = np.abs(i) / np.sqrt(1 - np.abs(o1_down) ** 2)
        over = test_data > 1.0
        under = test_data < 1.0
        print(f"Min t = {np.min(test_data):.5f}, Max t = {np.max(test_data):.5f}")
        print(f"t values > 1.0 = {over.sum()}, t values < 0.0 = {under.sum()}")
        print(f"Mean = {np.mean(test_data):.5f}, Median = {np.median(test_data):.5f}")
        test_hist, test_bins = np.histogram(test_data, T_BINS, T_RANGE, density=True)
        test_centers = 0.5 * (test_bins[1:] + test_bins[:-1])
        plt.plot(test_centers, test_hist, label=f"Index {k} over o1 down")
        k += 1
        print("-" * 100)
    plt.legend(loc="upper left")
    plot_file_name = f"qshe_t_dist_{n}.png"
    plt.savefig(plot_file_name, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file_name}")
    get_memory_usage("Overall memory usage for this analysis")
    print(
        f"Overall analysis done on {get_current_date()} after {time() - start_time:.3f} seconds"
    )
    print("=" * 100)
    output_file.close()
    sys.stdout = original_output
    print(
        f"Overall analysis done on {get_current_date()} after {time() - start_time:.3f} seconds"
    )
