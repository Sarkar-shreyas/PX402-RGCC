import numpy as np


def generate_phi():
    phi = np.random.uniform(0, 2 * np.pi, 4)
    return tuple(phi.tolist())


def generate_t():
    t = np.random.random(5)
    return tuple(t.tolist())


def generate_r(t1, t2, t3, t4, t5):
    r1 = np.sqrt(1 - t1**2)
    r2 = np.sqrt(1 - t2**2)
    r3 = np.sqrt(1 - t3**2)
    r4 = np.sqrt(1 - t4**2)
    r5 = np.sqrt(1 - t5**2)
    return tuple([r1, r2, r3, r4, r5])


# phi1, phi2, phi3, phi4 = phi.T
# t1, t2, t3, t4, t5 = t.T
# t1 = np.clip(t1, 0, 1)
# t2 = np.clip(t2, 0, 1)
# t3 = np.clip(t3, 0, 1)
# t4 = np.clip(t4, 0, 1)
# t5 = np.clip(t5, 0, 1)
# r1 = np.sqrt(1 - t1 * t1)
# r2 = np.sqrt(1 - t2 * t2)
# r3 = np.sqrt(1 - t3 * t3)
# r4 = np.sqrt(1 - t4 * t4)
# r5 = np.sqrt(1 - t5 * t5)


def calculate_t_prime(numerator, denominator):
    return np.abs(
        np.abs(numerator)
        / np.where(np.abs(denominator) < 1e-12, np.nan + 0j, denominator)
    )


if __name__ == "__main__":
    phi1, phi2, phi3, phi4 = generate_phi()
    t1, t2, t3, t4, t5 = generate_t()
    r1, r2, r3, r4, r5 = generate_r(t1, t2, t3, t4, t5)

    # Jack's Form from Shaw(Black)
    jack_numerator = (
        (r1 * t2)
        + ((r1**2 - t1**2) * (np.exp(1j * phi3)))
        - (t1 * t3 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3 - phi1)))
    )
    jack_denominator = (
        1
        - (t3 * t4 * t5 * np.exp(1j * phi4))
        - (r2 * r3 * r4 * np.exp(1j * (phi2)))
        - (t1 * t2 * t3 * t4 * np.exp(1j * (phi2 + phi4)))
        - (t1 * t2 * t3 * np.exp(1j * phi1))
        - (r1 * r3 * r5 * np.exp(1j * phi3))
        + (r1 * r2 * r3 * r4 * np.exp(1j * (phi2 + phi3)))
    )

    # My matrix (Blue)
    shreyas_numerator = (r1 * t2 * (1 - np.exp(1j * phi4) * t3 * t4 * t5)) - (
        np.exp(1j * phi3)
        * r5
        * (r3 * t2 + np.exp(1j * (phi2 - phi1)) * r2 * r4 * t1 * t3)
    )

    shreyas_denominator = (r3 - np.exp(1j * phi3) * r1 * r5) * (
        r3 - np.exp(1j * phi2) * r2 * r4
    ) + (t3 + np.exp(1j * phi4) * t4 * t5) * (t3 + np.exp(1j * phi1) * t1 * t2)

    # Shaw's form (2023 thesis paper)
    shaw_numerator = (
        -(np.exp(1j * (phi1 + phi4 - phi2)) * (r1 * r3 * r5 * t2 * t4))
        + ((t2 * t4) * (np.exp(1j * (phi1 + phi4))))
        - (np.exp(1j * phi4) * t1 * t3 * t4)
        + (np.exp(1j * phi3) * r2 * r3 * r4 * t1 * t5)
        - (np.exp(1j * phi1) * t2 * t3 * t5)
    )
    shaw_denominator = (
        -1
        - (r2 * r3 * r4 * np.exp(1j * (phi3)))
        + (r1 * r3 * r5 * np.exp(1j * phi2))
        + (r1 * r2 * r4 * r5 * np.exp(1j * (phi2 + phi3)))
        + (t1 * t2 * t3 * np.exp(1j * phi1))
        - (t1 * t2 * t4 * t5 * np.exp(1j * (phi1 + phi4)))
        + (t3 * t4 * t5 * np.exp(1j * phi4))
    )
    print("Phis: ", phi1, phi2, phi3, phi4)
    print("ts: ", t1, t2, t3, t4, t5)
    print("rs: ", r1, r2, r3, r4, r5)
    jack_t_prime = calculate_t_prime(jack_numerator, jack_denominator)
    shreyas_t_prime = calculate_t_prime(shreyas_numerator, shreyas_denominator)
    shaw_t_prime = calculate_t_prime(shaw_numerator, shaw_denominator)
    print(f"Jack: {jack_t_prime}\nShreyas: {shreyas_t_prime}\nShaw: {shaw_t_prime}")
