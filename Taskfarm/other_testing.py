import numpy as np
from source.utilities import (
    convert_t_to_z,
    launder,
    convert_z_to_t,
    generate_random_phases,
    generate_t_prime,
    extract_t_samples,
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ---------- Load data ---------- #
    data = np.load(
        "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm/v1.8S/EXP/shift_0.0/hist/z/z_hist_RG0.npz"
    )
    sym_data = np.load(
        "C:/Users/ssark/Desktop/Uni/Year 4 Courses/Physics Final Year Project/Project Code/Taskfarm/Data from taskfarm/v1.8S/FP/hist/sym_z/sym_z_hist_RG8.npz"
    )
    N = 10000000
    hist = data["histval"]
    bins = data["binedges"]
    centers = data["bincenters"]
    sym_hist = sym_data["histval"]
    sym_bins = sym_data["binedges"]
    sym_centers = sym_data["bincenters"]

    # ---------- Initial laundering ---------- #
    z = launder(N, hist, bins, centers)
    sym_z = launder(N, sym_hist, sym_bins, sym_centers)

    t = convert_z_to_t(z)
    sym_t = convert_z_to_t(sym_z)

    # ---------- t prime computation ---------- #
    phi = generate_random_phases(N)
    ts = extract_t_samples(sym_t, N)
    t_prime = generate_t_prime(ts, phi, "Shaw")
    print("Using Shaw's formula")

    # ---------- t data generation ---------- #
    t_hist, t_bins = np.histogram(t, 1000, range=(0, 1), density=True)
    t_centers = 0.5 * (t_bins[:-1] + t_bins[1:])
    sym_t_hist, sym_t_bins = np.histogram(sym_t, 1000, range=(0, 1), density=True)
    sym_t_centers = 0.5 * (sym_t_bins[:-1] + sym_t_bins[1:])
    t_prime_hist, t_prime_bins = np.histogram(t_prime, 1000, range=(0, 1), density=True)
    t_prime_centers = 0.5 * (t_prime_bins[:-1] + t_prime_bins[1:])
    t_launder = launder(N, sym_t_hist, sym_t_bins, sym_t_centers)

    t_launder_hist, t_launder_bins = np.histogram(
        t_launder, 1000, range=(0, 1), density=True
    )
    t_launder_centers = 0.5 * (t_launder_bins[:-1] + t_launder_bins[1:])

    # ---------- z data generation ---------- #
    z_hist, z_bins = np.histogram(z, 100000, (-25.0, 25.0), density=True)
    sym_z_hist, sym_z_bins = np.histogram(sym_z, 100000, (-25.0, 25.0), density=True)
    sym_z_centers = 0.5 * (sym_z_bins[:-1] + sym_z_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    z_prime = convert_t_to_z(t_prime)
    z_launder = convert_t_to_z(t_launder)
    z_prime_hist, z_prime_bins = np.histogram(
        z_prime, 100000, (-25.0, 25.0), density=True
    )
    z_prime_centers = 0.5 * (z_prime_bins[:-1] + z_prime_bins[1:])
    z_launder_hist, z_launder_bins = np.histogram(
        z_launder, 100000, (-25.0, 25.0), density=True
    )
    z_launder_centers = 0.5 * (z_launder_bins[:-1] + z_launder_bins[1:])
    # ---------- Tracking metrics ---------- #
    print(
        f"t prime over 1: {np.sum((t_prime > 1.0))}, t prime under 0: {np.sum((t_prime < 0.0))}\n Min: {np.min(t_prime)}, Max: {np.max(t_prime)}, Median: {np.median(t_prime)}"
    )
    print("-" * 100)
    print(
        f"t over 1: {np.sum((t > 1.0))}, t under 0: {np.sum((t < 0.0))}\n Min: {np.min(t)}, Max: {np.max(t)}, Median: {np.median(t)}"
    )
    print("-" * 100)
    print(
        f"sym t over 1: {np.sum((sym_t > 1.0))}, sym t under 0: {np.sum((sym_t < 0.0))}\n Min: {np.min(sym_t)}, Max: {np.max(sym_t)}, Median: {np.median(sym_t)}"
    )
    print("-" * 100)
    print(
        f"t launder over 1: {np.sum((t_launder > 1.0))}, t launder under 0: {np.sum((t_launder < 0.0))}\n Min: {np.min(t_launder)}, Max: {np.max(t_launder)}, Median: {np.median(t_launder)}"
    )
    print("-" * 100)
    print(
        f"z over 25: {np.sum((z > 25.0))}, z under -25: {np.sum((z < -25.0))}\n Min: {np.min(z)}, Max: {np.max(z)}, Median: {np.median(z)}"
    )
    print("-" * 100)
    print(
        f"sym z over 25: {np.sum((sym_z > 25.0))}, sym z under -25: {np.sum((sym_z < -25.0))}\n Min: {np.min(sym_z)}, Max: {np.max(sym_z)}, Median: {np.median(sym_z)}"
    )
    print("-" * 100)
    print(
        f"z prime over 25: {np.sum((z_prime > 25.0))}, z prime under -25: {np.sum((z_prime < -25.0))}\n Min: {np.min(z_prime)}, Max: {np.max(z_prime)}, Median: {np.median(z_prime)}"
    )
    print("-" * 100)
    print(
        f"z launder over 25: {np.sum((z_launder > 25.0))}, z launder under -25: {np.sum((z_launder < -25.0))}\n Min: {np.min(z_launder)}, Max: {np.max(z_launder)}, Median: {np.median(z_launder)}"
    )
    print("-" * 100)
    # ---------- Plotting ---------- #
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(t_centers, t_hist, label="t")
    ax0.plot(sym_t_centers, sym_t_hist, label="sym t")
    ax0.plot(t_prime_centers, t_prime_hist, label="t prime")
    ax0.plot(t_launder_centers, t_launder_hist, label="t launder")
    ax1.plot(z_centers, z_hist, label="z")
    ax1.plot(sym_z_centers, sym_z_hist, label="sym z")
    ax1.plot(z_prime_centers, z_prime_hist, label="z prime")
    ax1.plot(z_launder_centers, z_launder_hist, label="z launder")
    ax0.legend()
    ax1.legend()
    ax0.set_xlabel("t")
    ax0.set_ylabel("P(t)")
    ax0.set_title("Plots of t and t'")
    ax1.set_xlabel("z")
    ax1.set_ylabel("Q(z)")
    ax1.set_title("Plots of sym and unsym z")
    plt.show()
