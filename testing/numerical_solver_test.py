import numpy as np
from source.utilities import (
    convert_z_to_t,
    generate_initial_t_distribution,
    extract_t_samples,
    T_BINS,
    T_RANGE,
    convert_t_to_z,
    Z_BINS,
    Z_RANGE,
    launder,
    get_density,
    get_current_date,
)
from time import time
from constants import data_dir, CURRENT_VERSION, NUM_RG
import matplotlib.pyplot as plt


def generate_phases(n: int, i: int) -> np.ndarray:
    """
    Generate a (n, i) sized array of random phases
    """
    phi_sample = np.random.uniform(0, 2 * np.pi, (n, i))
    return phi_sample


def solve_single_A_and_b(
    ts: np.ndarray, phis: np.ndarray, batch_size: int = 100000
) -> np.ndarray:
    """
    Generates the A matrix from the given t, r and phi matrices
    """

    t1, t2, t3, t4, t5 = ts.T
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)
    phi12, phi15, phi23, phi31, phi34, phi42, phi45, phi53 = phis.T
    # fmt: off
    # A = np.array(
    #     [
    #         [1, 0, 0, 0, 0, -r1 * np.exp(1j * phi31), 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, t1 * np.exp(1j * phi31), 0, 0, 0, 0],
    #         [0, -t2 * np.exp(1j * phi12), 1, 0, 0, 0, 0, -r2 * np.exp(1j * phi42), 0, 0],
    #         [0, -r2 * np.exp(1j * phi12), 0, 1, 0, 0, 0, t2 * np.exp(1j * phi42), 0, 0],
    #         [0, 0, -r3 * np.exp(1j * phi23), 0, 1, 0, 0, 0, 0, -t3 * np.exp(1j * phi53)],
    #         [0, 0, t3 * np.exp(1j * phi23), 0, 0, 1, 0, 0, 0, -r3 * np.exp(1j * phi53)],
    #         [0, 0, 0, 0, t4 * np.exp(1j * phi34), 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, -r4 * np.exp(1j * phi34), 0, 0, 1, 0, 0],
    #         [-t5 * np.exp(1j * phi15), 0, 0, 0, 0, 0, -r5 * np.exp(1j * phi45), 0, 1, 0],
    #         [-r5 * np.exp(1j * phi15), 0, 0, 0, 0, 0, t5 * np.exp(1j * phi45), 0, 0, 1],
    #     ], dtype=np.complex128
    # )
    # b = np.array([t1,r1,0,0,0,0,0,0,0,0], dtype=np.complex128)
    # fmt: on
    # Initialise a batch-size array of A and b to do the solve in batches
    A = np.zeros((batch_size, 10, 10), dtype=np.complex128)
    b = np.zeros((batch_size, 10, 1), dtype=np.complex128)

    # Since it is initialised as a 3d array, we have to manually assign the values to indexes across every batch
    # Row 1
    A[:, 0, 0] = 1
    A[:, 0, 5] = -r1 * np.exp(1j * phi31)

    # Row 2
    A[:, 1, 1] = 1
    A[:, 1, 5] = t1 * np.exp(1j * phi31)

    # Row 3
    A[:, 2, 1] = -t2 * np.exp(1j * phi12)
    A[:, 2, 2] = 1
    A[:, 2, 7] = -r2 * np.exp(1j * phi42)

    # Row 4
    A[:, 3, 1] = -r2 * np.exp(1j * phi12)
    A[:, 3, 3] = 1
    A[:, 3, 7] = t2 * np.exp(1j * phi42)

    # Row 5
    A[:, 4, 2] = -r3 * np.exp(1j * phi23)
    A[:, 4, 4] = 1
    A[:, 4, 9] = -t3 * np.exp(1j * phi53)

    # Row 6
    A[:, 5, 2] = t3 * np.exp(1j * phi23)
    A[:, 5, 5] = 1
    A[:, 5, 9] = -r3 * np.exp(1j * phi53)

    # Row 7
    A[:, 6, 4] = t4 * np.exp(1j * phi34)
    A[:, 6, 6] = 1

    # Row 8
    A[:, 7, 4] = -r4 * np.exp(1j * phi34)
    A[:, 7, 7] = 1

    # Row 9
    A[:, 8, 0] = -t5 * np.exp(1j * phi15)
    A[:, 8, 6] = -r5 * np.exp(1j * phi45)
    A[:, 8, 8] = 1

    # Row 10
    A[:, 9, 0] = -r5 * np.exp(1j * phi15)
    A[:, 9, 6] = t5 * np.exp(1j * phi45)
    A[:, 9, 9] = 1

    # Assign b data
    b[:, 0, 0] = t1
    b[:, 1, 0] = r1

    x = np.linalg.solve(A, b)

    return x[:, 8]


def main(t_filename: str, z_filename: str):
    # Generate starting data
    start = time()
    n = 10000000
    i = 8
    t = generate_initial_t_distribution(n)
    ts = extract_t_samples(t, n)
    phis = generate_phases(n, i)
    print(f"Initial data generated on {get_current_date()}")
    # Use the numerical solver
    batch_size = 500000
    num_batches = n // batch_size
    print(
        f"Beginning numerical computation with {num_batches} batches of size {batch_size}"
    )
    output = np.empty(shape=(n, 1))
    for i in range(0, num_batches):
        output[i * batch_size : (i + 1) * batch_size] = np.abs(
            solve_single_A_and_b(
                ts[i * batch_size : (i + 1) * batch_size],
                phis[i * batch_size : (i + 1) * batch_size],
                batch_size,
            )
        )
        if i % 10 == 0:
            print(f"Completed the {i}th batch after {time() - start:.3f} seconds")
    # for k in range(n):
    #     output[k] = np.abs(solve_single_A_and_b(ts[k], phis[k]))
    #     if k % 500000 == 0:
    #         print(f"Currently at the {k}th step after {time() - start:.3f} seconds")

    print(f"Numerical solution computed within {time() - start:.3f} seconds")
    mask = output > 1.0
    print(f"{np.sum(mask)} values of output are above 1")
    output_hist, output_bins = np.histogram(output, T_BINS, T_RANGE, density=True)
    output_centers = 0.5 * (output_bins[1:] + output_bins[:-1])
    # Convert to z
    z_output = convert_t_to_z(output)
    z_output_hist, z_output_bins = np.histogram(z_output, Z_BINS, Z_RANGE, density=True)
    z_output_centers = 0.5 * (z_output_bins[:-1] + z_output_bins[1:])
    print(
        f"Arrays for numerical solution of t' and z generated after {time() - start:.3f} seconds"
    )
    # Load the existing FP distribution for t
    data = np.load(t_filename)
    loaded_hist = data["histval"]
    loaded_bins = data["binedges"]
    loaded_centers = data["bincenters"]
    loaded_density = get_density(loaded_hist, loaded_bins)
    # Load the existing FP distribution for z
    z_data = np.load(z_filename)
    z_hist = z_data["histval"]
    z_bin_edges = z_data["binedges"]
    z_centers = z_data["bincenters"]
    z_density = get_density(z_hist, z_bin_edges)

    # Sample from the FP
    sampled_z_data = launder(n, z_hist, z_bin_edges, z_centers)
    sampled_t_data = convert_z_to_t(sampled_z_data)
    sampled_t_hist, sampled_t_bins = np.histogram(
        sampled_t_data, T_BINS, T_RANGE, density=True
    )
    sampled_t_centers = 0.5 * (sampled_t_bins[1:] + sampled_t_bins[:-1])
    sampled_z_hist, sampled_z_bins = np.histogram(
        sampled_z_data, Z_BINS, Z_RANGE, density=True
    )
    sampled_z_centers = 0.5 * (sampled_z_bins[1:] + sampled_z_bins[:-1])

    fig, (ax0, ax1) = plt.subplots(1, 2, num="check", figsize=(10, 4))
    ax0.set_title(f"Numerical vs existing solution of t' form v{CURRENT_VERSION}")
    ax1.set_title(f"Numerical vs existing solution of z from v{CURRENT_VERSION}")
    ax0.plot(output_centers, output_hist, label="Numerical Solver", alpha=0.5)
    ax0.plot(loaded_centers, loaded_density, label="Existing FP")
    ax0.plot(sampled_t_centers, sampled_t_hist, label="Sampled from FP", alpha=0.5)
    ax1.plot(z_output_centers, z_output_hist, label="Numerical Solver", alpha=0.5)
    ax1.plot(z_centers, z_density, label="Existing FP")
    ax1.plot(sampled_z_centers, sampled_z_hist, label="Sampled from FP", alpha=0.5)
    ax1.set_xlim((-5.0, 5.0))
    ax0.legend(loc="upper left")
    ax1.legend(loc="upper right")
    plt.savefig(f"Numerical vs Existing for v{CURRENT_VERSION}.png", dpi=150)
    plt.close("check")
    print(f"Plots completed after {time() - start:.3f} seconds")


if __name__ == "__main__":
    t_filename = (
        f"{data_dir}/v{CURRENT_VERSION}/FP/hist/input_t/input_t_hist_RG{NUM_RG - 1}.npz"
    )
    z_filename = (
        f"{data_dir}/v{CURRENT_VERSION}/FP/hist/sym_z/sym_z_hist_RG{NUM_RG - 1}.npz"
    )
    main(t_filename, z_filename)
    print(f"Analysis completed on {get_current_date()}")
