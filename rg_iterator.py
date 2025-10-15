import numpy as np
import matplotlib
from typing import Optional

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from utils import convert_t_to_z, convert_z_to_g
from distribution_production import (
    generate_initial_t_distribution,
    generate_random_phases,
    center_z_distribution,
    Probability_Distribution,
    extract_t_samples,
)


# ---------- t prime definition ---------- #
def generate_t_prime(t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Takes in an array of sample t and phi values, and returns the computed t' value.

    Args:
        t (np.ndarray): A numpy array of t values in the shape (N, 5) where N is the number of iterations
        phi (np.ndarray): A numpy array of phi values in the shape (N, 4) where N is the number of iterations

    Returns:
        t' (np.ndarray): A numpy array of t' values calculated based on input t and phi values, with shape (N,) where N is the number of iterations
    """
    phi1, phi2, phi3, phi4 = phi.T
    t1, t2, t3, t4, t5 = t.T
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)

    numerator = (r1 * t2 * (1 - np.exp(1j * phi4) * t3 * t4 * t5)) - (
        np.exp(1j * phi3)
        * r5
        * (r3 * t2 + np.exp(1j * (phi2 - phi1)) * r2 * r4 * t1 * t3)
    )

    denominator = (r3 - np.exp(1j * phi3) * r1 * r5) * (
        r3 - np.exp(1j * phi2) * r2 * r4
    ) + (t3 + np.exp(1j * phi4) * t4 * t5) * (t3 + np.exp(1j * phi1) * t1 * t2)

    t_prime = np.abs(
        numerator / np.where(np.abs(denominator) < 1e-12, np.nan + 0j, denominator)
    )

    return t_prime


# ---------- RG Factory ---------- #
def rg_iterations(
    N: int,
    bins: int,
    K: int,
    existing_distribution: Optional[Probability_Distribution] = None,
) -> tuple[Probability_Distribution, Probability_Distribution, list]:
    """
    Main factory that performs the RG steps until a fixed point distribution is obtained
    Performs a maximum of K steps
    """

    # Generate initial P(t) = 2t distribution
    if not existing_distribution:
        initial_t = generate_initial_t_distribution(N)
        P_t = Probability_Distribution(initial_t, bins)
    else:
        P_t = existing_distribution

    # Setup variables for iteration and storage
    previous_Qz: Probability_Distribution | None = None
    parameter_storage = []

    plt.figure(figsize=(7, 4))
    plt.xlim([-5, 5])
    plt.ylim([0.0, 0.27])
    plt.xlabel("z")
    plt.ylabel("Q(z)")
    plt.title("Evolution of Q(z)")

    inner_start_time = time.time()
    print("-" * 100)
    print("Beginning procedure")

    # RG procedure, breaks early if convergence reached
    for _ in range(K):
        current_time = time.time()
        print(
            f"At iteration {_}. It has been {current_time - inner_start_time:.3f} seconds since entering the loop"
        )

        # Generate t and phi samples with updated distributions
        t_sample = extract_t_samples(P_t, N)
        phi_sample = generate_random_phases(N)

        # Generate t' and z distribution values
        next_t = generate_t_prime(t_sample, phi_sample)
        next_z = convert_t_to_z(next_t)
        print(f"t and z have been generated for iteration {_}")

        # Recenter z and initialise histogram
        next_z = center_z_distribution(next_z)
        current_Qz = Probability_Distribution(next_z, bins)
        if _ in set(range(1, K, 7)):
            centers = 0.5 * (current_Qz.bin_edges[:-1] + current_Qz.bin_edges[1:])
            plt.plot(centers, current_Qz.histogram_values, label=f"Iteration {_}")
            print(f"Values have been plotted for iteration {_}")

        # Check for convergence
        if previous_Qz is not None:
            delta = current_Qz.histogram_distances(
                previous_Qz.histogram_values, previous_Qz.bin_edges
            )
            previous_mean, previous_std = previous_Qz.mean_and_std()
            current_mean, current_std = current_Qz.mean_and_std()
            parameter_storage.append((_, delta, current_std))
            if delta < 1e-3:
                print(f"Converged at iteration #{_}")
                if _ >= 7:
                    plt.savefig("z_dist.png", dpi=150)
                    return current_Qz, P_t, parameter_storage
            else:
                print(f"Didn't converge in iteration {_}, onto the next.")
                print("=" * 100)

        # Update distributions and values for next iteration

        next_g = convert_z_to_g(current_Qz.sample(N))
        next_t = np.sqrt(next_g)
        P_t = Probability_Distribution(next_t, bins)
        previous_Qz = current_Qz

    # If it didn't converge, return the final set of data
    plt.legend()

    plt.savefig("plots/z_dist.png", dpi=150)
    return previous_Qz, P_t, parameter_storage  # type: ignore
