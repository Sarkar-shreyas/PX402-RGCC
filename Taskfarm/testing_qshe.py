# flake8: noqa: E501
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

process = psutil.Process(os.getpid())


def get_memory_usage(item: str = "") -> None:
    """Prints the memory usage for a process"""
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"{item} = {memory:.3f} MB")


def solve_qshe_matrix_eq(
    ts: np.ndarray, taus: np.ndarray, phis: np.ndarray, batch_size: int
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

    # Row 0
    M[:, 0, 0] = 1
    M[:, 0, 9] = -1j * t1 * tau1 * np.exp(1j * phi31)
    M[:, 0, 19] = -f1 * np.exp(1j * phi51)

    # Row 1
    M[:, 1, 1] = 1
    M[:, 1, 7] = f1 * np.exp(1j * phi21)
    M[:, 1, 9] = -r1 * tau1 * np.exp(1j * phi31)

    # Row 2
    M[:, 2, 2] = 1
    M[:, 2, 7] = -1j * t1 * tau1 * np.exp(1j * phi21)
    M[:, 2, 19] = -r1 * tau1 * np.exp(1j * phi51)

    # Row 3
    M[:, 3, 3] = 1
    M[:, 3, 7] = -r1 * tau1 * np.exp(1j * phi21)
    M[:, 3, 9] = -f1 * np.exp(1j * phi31)
    M[:, 3, 19] = -1j * t1 * tau1 * np.exp(1j * phi51)

    # Row 4
    M[:, 4, 0] = -r2 * tau2 * np.exp(1j * phi12)
    M[:, 4, 4] = 1
    M[:, 4, 11] = -f2 * np.exp(1j * phi32)
    M[:, 4, 12] = -1j * t2 * tau2 * np.exp(1j * phi42)

    # Row 5
    M[:, 5, 0] = -1j * t2 * tau2 * np.exp(1j * phi12)
    M[:, 5, 5] = 1
    M[:, 5, 12] = -r2 * tau2 * np.exp(1j * phi42)

    # Row 6
    M[:, 6, 0] = f2 * np.exp(1j * phi12)
    M[:, 6, 6] = 1
    M[:, 6, 11] = -r2 * tau2 * np.exp(1j * phi32)

    # Row 7
    M[:, 7, 7] = 1
    M[:, 7, 11] = -1j * t2 * tau2 * np.exp(1j * phi32)
    M[:, 7, 12] = -f2 * np.exp(1j * phi42)

    # Row 8
    M[:, 8, 2] = -f3 * np.exp(1j * phi13)
    M[:, 8, 5] = -r3 * tau3 * np.exp(1j * phi23)
    M[:, 8, 8] = 1
    M[:, 8, 16] = -1j * t3 * tau3 * np.exp(1j * phi53)

    # Row 9
    M[:, 9, 5] = -1j * t3 * tau3 * np.exp(1j * phi23)
    M[:, 9, 9] = 1
    M[:, 9, 15] = f3 * np.exp(1j * phi43)
    M[:, 9, 16] = -r3 * tau3 * np.exp(1j * phi53)

    # Row 10
    M[:, 10, 2] = -r3 * tau3 * np.exp(1j * phi13)
    M[:, 10, 5] = f3 * np.exp(1j * phi23)
    M[:, 10, 10] = 1
    M[:, 10, 15] = -1j * t3 * tau3 * np.exp(1j * phi43)

    # Row 11
    M[:, 11, 2] = -1j * t3 * tau3 * np.exp(1j * phi13)
    M[:, 11, 11] = 1
    M[:, 11, 15] = -r3 * tau3 * np.exp(1j * phi43)
    M[:, 11, 16] = -f3 * np.exp(1j * phi53)

    # Row 12
    M[:, 12, 8] = -r4 * tau4 * np.exp(1j * phi34)
    M[:, 12, 12] = 1
    M[:, 12, 17] = -f4 * np.exp(1j * phi54)

    # Row 13
    M[:, 13, 6] = f4 * np.exp(1j * phi24)
    M[:, 13, 8] = -1j * t4 * tau4 * np.exp(1j * phi34)
    M[:, 13, 13] = 1

    # Row 14
    M[:, 14, 6] = -1j * t4 * tau4 * np.exp(1j * phi24)
    M[:, 14, 8] = f4 * np.exp(1j * phi34)
    M[:, 14, 14] = 1
    M[:, 14, 18] = -r4 * tau4 * np.exp(1j * phi54)

    # Row 15
    M[:, 15, 6] = -r4 * tau4 * np.exp(1j * phi24)
    M[:, 15, 15] = 1
    M[:, 15, 18] = -1j * t4 * tau4 * np.exp(1j * phi54)

    # Row 16
    M[:, 16, 1] = -r5 * tau5 * np.exp(1j * phi15)
    M[:, 16, 13] = -1j * t5 * tau5 * np.exp(1j * phi45)
    M[:, 16, 16] = 1

    # Row 17
    M[:, 17, 1] = -1j * t5 * tau5 * np.exp(1j * phi15)
    M[:, 17, 10] = f5 * np.exp(1j * phi35)
    M[:, 17, 13] = -r5 * tau5 * np.exp(1j * phi45)
    M[:, 17, 17] = 1

    # Row 18
    M[:, 18, 1] = f5 * np.exp(1j * phi15)
    M[:, 18, 10] = -1j * t5 * tau5 * np.exp(1j * phi35)
    M[:, 18, 18] = 1

    # Row 19
    M[:, 19, 10] = -r5 * tau5 * np.exp(1j * phi35)
    M[:, 19, 13] = -f5 * np.exp(1j * phi45)
    M[:, 19, 19] = 1

    # b matrix. Set I1 = 1, I2 = 0, I3 = 0, I4 = 0. i.e we only have a spin up electron entering the 1st node.
    b[:, 0, 0] = r1 * tau1
    b[:, 1, 0] = 1j * t1 * tau1
    b[:, 2, 0] = -f1

    x = np.linalg.solve(M, b)

    # Corresponds to spin-up electron entering I1 and exiting at output O5* (also spin up)
    # return x[:, 17]
    return x


def numerical_solver(
    ts: np.ndarray, taus: np.ndarray, phis: np.ndarray, N: int
) -> np.ndarray:
    """Solve the matrix equation Mx=b for N samples using batching"""
    start = time()
    num_batches = 50
    batch_size = N // num_batches
    output = np.empty(shape=(N, 20, 1), dtype=np.float64)
    print(f"Beginning numerical solver on {get_current_date()}")
    get_memory_usage("Memory usage before computation")
    for i in range(num_batches):
        indexes = slice(i * batch_size, (i + 1) * batch_size)
        output[indexes, :] = np.abs(
            solve_qshe_matrix_eq(ts[indexes], taus[indexes], phis[indexes], batch_size)
        )
        if (i + 1) % 10 == 0:
            get_memory_usage(f"Memory usage after batch {i + 1}")
            print(f"Batch {i + 1} done in {time() - start:.3f} seconds")
    print("-" * 100)
    print(
        f"Computation for all {num_batches} batches done after {time() - start:.3f} seconds"
    )
    return output


if __name__ == "__main__":
    start_time = time()
    n = int(5e6)
    num_phases = 16
    print(f"Program started on {get_current_date()}")
    print(f"Starting numerical solver for QSHE matrix with {n} samples")
    print("-" * 100)
    # Generate initial arrays
    get_memory_usage("Memory usage before generating initial data")
    starting_t = generate_initial_t_distribution(n)
    starting_tau = generate_initial_t_distribution(n)
    # starting_tau = np.random.uniform(0, 1, n)
    phases = generate_random_phases(n, num_phases)
    t_samples = extract_t_samples(starting_t, n)
    tau_samples = extract_t_samples(starting_tau, n)
    print(
        f"Initial t, tau and phi arrays generated after {time() - start_time:.3f} seconds"
    )
    get_memory_usage("Memory usage after generating initial data")
    print("-" * 100)
    # Run the solver
    data = numerical_solver(t_samples, tau_samples, phases, n)
    print(f"Data returned with shape {data.shape}")
    get_memory_usage("Memory usage after solving for all outputs")
    print("-" * 100)
    plt.figure(figsize=(24, 12))
    plt.title("Distribution of t")
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.ylim((0, 5))
    starting_hist, starting_bins = np.histogram(
        starting_t, T_BINS, T_RANGE, density=True
    )
    plt.plot(
        (0.5 * (starting_bins[1:] + starting_bins[:-1])), starting_hist, label="Initial"
    )
    for i in range(20):
        print(f"Printing stats and plotting data for output row {i + 1}")
        # Get some statistics
        over_mask = data[:, i] > 1.0
        under_mask = data[:, i] < 0.0
        print(f"Min t = {np.min(data[:, i]):.5f}, Max t = {np.max(data[:, i]):.5f}")
        print(
            f"t values > 1.0 = {over_mask.sum()}, t values < 0.0 = {under_mask.sum()}"
        )
        print(f"Mean = {np.mean(data[:, i]):.5f}, Median = {np.median(data[:, i]):.5f}")
        # Plot the data
        hist, bins = np.histogram(data[:, i], T_BINS, T_RANGE, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=f"Row {i + 1}")
        print("-" * 100)

    plt.legend(loc="upper left")
    plt.savefig("Distribution of t for qshe.png", dpi=150)
    plt.close()
    print(
        f"Overall analysis done on {get_current_date()} after {time() - start_time:.3f} seconds"
    )
    get_memory_usage("Overall memory usage for this analysis")
    print("=" * 100)
