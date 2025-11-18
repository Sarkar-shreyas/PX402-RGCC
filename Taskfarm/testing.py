import numpy as np
from source.utilities import (
    extract_t_samples,
    generate_initial_t_distribution,
    generate_random_phases,
    convert_t_to_z,
    launder,
    convert_z_to_t,
    center_z_distribution,
)
import matplotlib.pyplot as plt

t_tol = np.sqrt(1 / (1 + np.exp(25.0)))


def generate_t_prime(t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    phi1, phi2, phi3, phi4 = phi.T
    t1, t2, t3, t4, t5 = t.T

    # t1 = np.clip(t1, t_tol, 1 - t_tol)
    # t2 = np.clip(t2, t_tol, 1 - t_tol)
    # t3 = np.clip(t3, t_tol, 1 - t_tol)
    # t4 = np.clip(t4, t_tol, 1 - t_tol)
    # t5 = np.clip(t5, t_tol, 1 - t_tol)
    r1 = np.sqrt(1 - (t1 * t1))
    r2 = np.sqrt(1 - (t2 * t2))
    r3 = np.sqrt(1 - (t3 * t3))
    r4 = np.sqrt(1 - (t4 * t4))
    r5 = np.sqrt(1 - (t5 * t5))
    ts = [t1, t2, t3, t4, t5]
    rs = [r1, r2, r3, r4, r5]

    # Shaw's form (2023 thesis paper)
    numerator = (
        -(np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
        + ((t2 * t4) * (np.exp(1j * (phi1 + phi4))))
        - (np.exp(1j * phi4) * t1 * t3 * t4)
        + (np.exp(1j * phi3) * r2 * r3 * r4 * t1 * t5)
        - (np.exp(1j * phi1) * t2 * t3 * t5)
    )
    denominator = (
        -1
        - (r2 * r3 * r4 * np.exp(1j * (phi3)))
        + (r1 * r3 * r5 * np.exp(1j * phi2))
        + (r1 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3)))
        + (t1 * t2 * t3 * np.exp(1j * phi1))
        - (t1 * t2 * t4 * t5 * np.exp(1j * (phi1 + phi4)))
        + (t3 * t4 * t5 * np.exp(1j * phi4))
    )

    # t_prime = np.abs(
    #     numerator / np.where(np.abs(denominator) < 1e-12, np.nan + 0j, denominator)
    # )
    t_prime = np.abs(numerator) / np.abs(denominator)
    # t_prime = np.abs(numerator / denominator)
    over_mask = t_prime > 1.0
    under_mask = t_prime < 0.0
    tp_over = np.where(over_mask)[0]
    tp_under = np.where(under_mask)[0]
    num_over = numerator[tp_over]
    den_over = denominator[tp_over]

    print(
        f"There are {tp_over.size} values of tprime greater than 1, and {tp_under.size} values of tprime smaller than 0"
    )
    print(
        f"Tpover: Mean = {np.mean(t_prime[tp_over])}, Median = {np.median(t_prime[tp_over])}, Min = {np.min(t_prime[tp_over])}, Max = {np.max(t_prime[tp_over])}"
    )
    print(
        f"Numerator of tpover: Min = {np.min(np.abs(num_over))}, Max = {np.max(np.abs(num_over))}, Median = {np.median(np.abs(num_over))}"
    )
    print(
        f"Denominator of tpover: Min = {np.min(np.abs(den_over))}, Max = {np.max(np.abs(den_over))}, Median = {np.median(np.abs(den_over))}"
    )
    print(
        f"Overall Numerator: Min = {np.min(np.abs(numerator))}, Max = {np.max(np.abs(numerator))}, Median = {np.median(np.abs(numerator))}"
    )
    print(
        f"Overall Denominator: Min = {np.min(np.abs(denominator))}, Max = {np.max(np.abs(denominator))}, Median = {np.median(np.abs(denominator))}"
    )
    for i in range(5):
        t_over = ts[i][over_mask]
        r_over = rs[i][over_mask]
        # t_under = ts[i][under_mask]
        # r_under = rs[i][under_mask]
        print(
            f"Var {i + 1} - Over: t = {t_over.size}, Mean of t = {np.mean(t_over)}, Median of t = {np.median(t_over)}, Max of t = {np.max(t_over)}"
        )
        print(
            f"Var {i + 1} - Over: r = {r_over.size}, Mean of r = {np.mean(r_over)}, Median of r = {np.median(r_over)}, Max of r = {np.max(r_over)}"
        )

    return t_prime
    # return t_prime[np.isfinite(t_prime)]
    # return np.clip(t_prime, t_tol, 1 - t_tol)


if __name__ == "__main__":
    N = 1000000
    phi = generate_random_phases(N)
    t = generate_initial_t_distribution(N)
    ts = extract_t_samples(t, N)
    t_prime = generate_t_prime(ts, phi)
    hist, bins = np.histogram(t_prime, 1000, (0, 1), density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    # tp_over = np.sum(t_prime > (1.0 - t_tol))
    # tp_under = np.sum(t_prime < t_tol)
    z = convert_t_to_z(t_prime)
    z_hist, z_bins = np.histogram(z, 100000, (-50.0, 50.0), density=True)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    print(f"Finite z = {np.sum(np.isfinite(z))}")
    sym_z = center_z_distribution(z_hist, z_bins)
    sym_hist, sym_bins = np.histogram(sym_z, z_bins, density=True)
    laundered_t = convert_z_to_t(launder(N, sym_z, z_bins, z_centers))
    # laundered_t = convert_z_to_t(launder(N, z_hist, z_bins, z_centers))
    # over = np.sum(laundered_t > (1.0 - t_tol))
    # under = np.sum(laundered_t < t_tol)
    launder_hist, launder_bins = np.histogram(laundered_t, 1000, (0, 1), density=True)
    launder_centers = 0.5 * (launder_bins[:-1] + launder_bins[1:])
    plt.figure("t")
    plt.plot(centers, hist)
    plt.savefig("testing_t.png", dpi=150)
    plt.close("t")
    plt.figure("z")
    plt.scatter(z_centers[::10], z_hist[::10], label="Base")
    plt.plot(z_centers, sym_hist, label="Sym")
    plt.legend(loc="upper left")
    plt.xlim((-5.0, 5.0))
    plt.savefig("testing_z.png", dpi=150)
    plt.close("z")
    plt.figure("launder")
    plt.plot(launder_centers, launder_hist)
    plt.savefig("testing_launder.png", dpi=150)
    plt.close("launder")
