import numpy as np
import matplotlib
from typing import Optional
from mpl_toolkits.axes_grid1 import inset_locator
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from .utils import convert_t_to_z, convert_z_to_g
from .distribution_production import (
    generate_initial_t_distribution,
    generate_random_phases,
    center_z_distribution,
    Probability_Distribution,
    extract_t_samples,
)
from config import BINS, N, T_RANGE, EXPRESSION, DIST_TOLERANCE, STD_TOLERANCE


# ---------- t prime definition ---------- #
def generate_t_prime(t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Generate next-step amplitudes using the RG transformation.

    Implements the core RG transformation that maps five input amplitudes and
    four phases to a new amplitude t'. The transformation preserves important
    symmetries while capturing the essential physics of the model.

    Parameters
    ----------
    t : numpy.ndarray
        Array of shape (N, 5) containing five amplitude samples per row.
        Values should be in range [0,1].
    phi : numpy.ndarray
        Array of shape (N, 4) containing four random phases per row.
        Values should be in range [0,2π].

    Returns
    -------
    numpy.ndarray
        Array of shape (N,) containing the transformed amplitudes t'.
        Values are clipped to range [0,1-1e-15] for numerical stability.

    Notes
    -----
    The transformation includes safeguards against division by zero and
    produces values strictly less than 1 to prevent numerical issues in
    subsequent logarithmic transformations.
    """
    phi1, phi2, phi3, phi4 = phi.T
    t1, t2, t3, t4, t5 = t.T
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)
    t3 = np.clip(t3, 0, 1)
    t4 = np.clip(t4, 0, 1)
    t5 = np.clip(t5, 0, 1)
    r1 = np.sqrt(1 - t1 * t1)
    r2 = np.sqrt(1 - t2 * t2)
    r3 = np.sqrt(1 - t3 * t3)
    r4 = np.sqrt(1 - t4 * t4)
    r5 = np.sqrt(1 - t5 * t5)

    if EXPRESSION == "Jack":
        # Jack's Form from Shaw(Black)
        numerator = (
            (r1 * t2)
            + ((r1**2 - t1**2) * (np.exp(1j * phi3)))
            - (t1 * t3 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3 - phi1)))
        )
        denominator = (
            1
            - (t3 * t4 * t5 * np.exp(1j * phi4))
            - (r2 * r3 * r4 * np.exp(1j * (phi2)))
            - (t1 * t2 * t3 * t4 * np.exp(1j * (phi2 + phi4)))
            - (t1 * t2 * t3 * np.exp(1j * phi1))
            - (r1 * r3 * r5 * np.exp(1j * phi3))
            + (r1 * r2 * r3 * r4 * np.exp(1j * (phi2 + phi3)))
        )
    elif EXPRESSION == "Shreyas":
        # My matrix (Blue)
        numerator = (r1 * t2 * (1 - np.exp(1j * phi4) * t3 * t4 * t5)) - (
            np.exp(1j * phi3)
            * r5
            * (r3 * t2 + np.exp(1j * (phi2 - phi1)) * r2 * r4 * t1 * t3)
        )

        denominator = (r3 - np.exp(1j * phi3) * r1 * r5) * (
            r3 - np.exp(1j * phi2) * r2 * r4
        ) + (t3 + np.exp(1j * phi4) * t4 * t5) * (t3 + np.exp(1j * phi1) * t1 * t2)
    else:
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
    t_prime = np.abs(np.abs(numerator) / np.abs(denominator))
    return t_prime
    return np.clip(t_prime, 0.0, 1.0 - 1e-15)


# ---------- RG Factories ---------- #
def rg_iterations_for_fp(
    N: int,
    bins: int,
    K: int,
    existing_distribution: Optional[Probability_Distribution] = None,
) -> tuple[Probability_Distribution, Probability_Distribution, list]:
    """Perform repeated RG transformations to find the fixed point distribution.

    This function implements the iterative RG procedure to find the fixed point
    distribution Q*(z). Starting from either a provided distribution or the
    default P(t)=2t, it applies the RG transformation repeatedly until either
    convergence is detected or the maximum number of steps is reached.

    The convergence criterion is based on the L2 distance between successive
    distributions being less than 1e-3 for three consecutive iterations.

    Parameters
    ----------
    N : int
        Number of samples to use in each iteration.
    bins : int
        Number of bins for histogramming the distributions.
    K : int
        Maximum number of RG steps to perform.
    existing_distribution : Probability_Distribution, optional
        Starting distribution. If None, uses P(t)=2t.

    Returns
    -------
    tuple[Probability_Distribution, Probability_Distribution, list]
        A tuple containing:
        - Q*(z): The (approximately) fixed point z-distribution
        - P*(t): The corresponding t-distribution
        - params: List of (step, distance, std) tuples tracking convergence

    Notes
    -----
    The function also generates plots showing the evolution of both Q(z)
    and P(t) distributions, saved to the plots directory.
    """

    # Generate initial P(t) = 2t distribution
    if not existing_distribution:
        initial_t = generate_initial_t_distribution(N)
        P_t = Probability_Distribution(initial_t, bins, range=T_RANGE)
    else:
        P_t = existing_distribution

    # Setup variables for iteration and storage
    previous_Qz: Probability_Distribution | None = None
    parameter_storage = []
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(0.0, 0.3)
    ax0.set_xlabel("z")
    ax0.set_ylabel("Q(z)")
    ax0.set_title("Evolution of Q(z)")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.0, 4)
    ax1.set_xlabel("t")
    ax1.set_ylabel("P(t)")
    ax1.set_title("Evolution of P(t)")
    ax2 = inset_locator.inset_axes(ax0, width="30%", height=0.6)
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(0, 0.3)
    ax2.set_xlabel("z")
    ax2.set_ylabel("Q(z)")
    inner_start_time = time.time()
    print("-" * 100)
    print("Beginning procedure")
    num_convergences = 0
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
        current_Qz = Probability_Distribution(next_z, bins)
        current_Qz = center_z_distribution(current_Qz)
        if _ in set([1, 2, K - 1]):
            z_centers = 0.5 * (current_Qz.bin_edges[:-1] + current_Qz.bin_edges[1:])
            ax0.plot(z_centers, current_Qz.histogram_values, label=f"Iteration {_}")
            t_centers = 0.5 * (P_t.bin_edges[:-1] + P_t.bin_edges[1:])
            ax1.plot(t_centers, P_t.histogram_values, label=f"Iteration {_}")
            ax2.plot(z_centers, current_Qz.histogram_values)
            print(f"Values have been plotted for iteration {_}")

        # Check for convergence
        if previous_Qz is not None:
            delta = current_Qz.histogram_distances(
                previous_Qz.histogram_values, previous_Qz.bin_edges
            )
            previous_mean, previous_std = previous_Qz.mean_and_std()
            current_mean, current_std = current_Qz.mean_and_std()
            std_diff = current_std - previous_std
            parameter_storage.append((_, delta, current_std, current_mean))
            if delta < DIST_TOLERANCE:
                if std_diff <= STD_TOLERANCE:
                    print(f"std converged as well: {std_diff} at step {_}")
                num_convergences += 1
                print(f"Converged at iteration #{_}")
                if num_convergences == 3 or _ == K - 1:
                    np.savetxt(
                        f"data/{EXPRESSION}/Fixed_point/Converged_t_prime_{N}_iters.txt",
                        next_t,
                    )
                    ax0.legend()
                    ax1.legend()
                    plt.savefig(
                        f"plots/{EXPRESSION}/Fixed_point/Converged_z_dist_with_{N}_iters.png",
                        dpi=150,
                    )
                    print("Updated plot file with FP distribution")
                    print("-" * 100)
                    return current_Qz, P_t, parameter_storage
            else:
                print(f"Didn't converge in iteration {_}, onto the next.")
                print("=" * 100)

        # Update distributions and values for next iteration

        next_g = convert_z_to_g(current_Qz.sample(N))
        next_t = np.sqrt(next_g)
        P_t = Probability_Distribution(next_t, bins, range=T_RANGE)
        previous_Qz = current_Qz

    # If it didn't converge, return the final set of data
    np.savetxt(f"data/{EXPRESSION}/Fixed_point/t_prime_{N}_iters.txt", next_t)
    np.savetxt(
        f"data/{EXPRESSION}/Fixed_point/p_t_{N}_iters.txt",
        P_t.histogram_values,
    )
    np.savetxt(
        f"data/{EXPRESSION}/Fixed_point/p_t_bins_{N}_iters.txt",
        P_t.bin_edges,
    )
    np.savetxt(
        f"data/{EXPRESSION}/Fixed_point/q_z_{N}_iters.txt",
        previous_Qz.histogram_values,  # type: ignore
    )
    np.savetxt(
        f"data/{EXPRESSION}/Fixed_point/q_z_bins_{N}_iters.txt",
        previous_Qz.bin_edges,  # type: ignore
    )
    gauss_z = norm.pdf(previous_Qz.sample(N), -25, 25)  # type: ignore
    x = np.linspace(-25, 25, N)
    # ax0.plot(x, gauss_z, marker="o", color="green")
    ax0.legend(loc="upper left")
    ax1.legend()
    # plt.tight_layout()
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    print("Updated plot file without convergence")
    print("-" * 100)
    plt.savefig(f"plots/{EXPRESSION}/Fixed_point/z_dist_with_{N}_iters.png", dpi=150)
    plt.close()
    return previous_Qz, P_t, parameter_storage  # type: ignore


def rg_iterator_for_nu(Qz: Probability_Distribution) -> Probability_Distribution:
    """Perform a single RG transformation step for critical exponent analysis.

    This variant of the RG transformation is specifically designed for critical
    exponent estimation. Unlike the fixed point iterator, it does not recenter
    the distribution after transformation, preserving the absolute position of
    peaks needed for tracking how perturbations evolve.

    Parameters
    ----------
    Qz : Probability_Distribution
        Current Q(z) distribution, typically a perturbed version of Q*(z).

    Returns
    -------
    Probability_Distribution
        The transformed distribution Q_{k+1}(z) without recentering.

    Notes
    -----
    This function maintains the same bin structure as the input distribution
    and uses the same sampling parameters N defined in config.
    """

    z_sample = Qz.sample(N)
    g_values = convert_z_to_g(z_sample)
    t_values = np.sqrt(g_values)
    P_t = Probability_Distribution(t_values, BINS, range=T_RANGE)
    t_sample = extract_t_samples(P_t, N)
    phi_samples = generate_random_phases(N)
    t_prime = generate_t_prime(t_sample, phi_samples)
    next_z = convert_t_to_z(t_prime)
    next_Qz = Probability_Distribution(next_z, BINS)

    return next_Qz
