"""Run an RG workflow locally.

This script provides a single-process driver used for local testing of the RG
Monte Carlo pipeline. It re-uses the library code in ``source/`` to run small
FP (fixed-point) or EXP (shifted/exponent) workflows and writes NPZ histograms
and a JSON manifest into a local output directory.

Usage
-----
Run from the repository root::

        python -m Local.run_local --config Local/configs/local_iqhe --set "rg_settings.steps=2" --set "rg_settings.samples=10000" --type FP

Notes
-----
- This module redirects stdout/stderr to ``output.txt``/``error.txt`` inside
    the chosen output directory when executed as a script.
"""

from time import time
import numpy as np
from numpy.typing import ArrayLike
import sys
import json
from pathlib import Path
from source.parse_config import build_parser, get_project_root, validate_input
from source.config import (
    handle_config,
    save_updated_config,
    get_nested_data,
    build_config,
    RGConfig,
)
from source.utilities import (
    convert_t_to_z,
    convert_z_to_t,
    generate_constant_array,
    extract_t_samples,
    # generate_initial_t_distribution,
    generate_random_phases,
    get_current_date,
    rg_data_workflow,
    build_rng,
    center_z_distribution,
    launder,
    get_density,
    save_data,
)
from QSHE.testing_qshe import (
    # solve_qshe_matrix_eq,
    numerical_solver,
    gen_initial_data,
    append_parser,
)
# from constants import T_DICT, PHI_DICT

# ---------- Helper utility ---------- #


def build_default_output_dir(config: dict) -> Path:
    """Build the default local output directory for a config.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary (result of :func:`source.parse_config.validate_input`
        / :func:`source.config.handle_config`). Must include ``main.version`` and
        ``engine.method`` keys. ``engine.expr`` is also used to form the directory
        name.

    Returns
    -------
    Path
        A path under the repository root of the form
        ``<repo_root>/Local data/{version}_{method}_{expr}``.
    """
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root(1)

    return root / "Local data" / version_str


def build_hist(data: np.ndarray, bins: int, range: tuple) -> dict:
    """Compute a histogram and return related arrays and densities.

    Parameters
    ----------
    data : np.ndarray
        1-D array of samples to histogram.
    bins : int
        Number of histogram bins.
    range : tuple
        (min, max) binning range.

    Returns
    -------
    dict
        Dictionary with keys: ``hist`` (counts), ``edges`` (bin edges),
        ``centers`` (bin centers) and ``densities`` (density per bin computed
        using :func:`source.utilities.get_density`).
    """
    hist, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    densities = get_density(hist, edges)
    return {"hist": hist, "edges": edges, "centers": centers, "densities": densities}


def print_config(config: RGConfig) -> None:
    """Print a compact, human-readable summary of the main run settings.

    Parameters
    ----------
    config : RGConfig
        Configuration dataclass returned by :func:`source.config.build_config`.

    Notes
    -----
    - If the configuration indicates an ``EXP`` run, this function expects
      ``config.shifts`` to be iterable.
    """
    header = f" RG Configuration for {config.version}_{config.method}_{config.expr} "
    print(header)
    print("-" * len(header))
    width = 18

    def p(k, v):
        print(f"{k:{width}}: {v}")

    p("Total samples", config.samples)
    p("Steps", config.steps)
    p("Seed", config.seed)
    p("Symmetrising", bool(config.symmetrise))
    p("Type", getattr(config, "type", ""))
    if config.type.strip().upper() == "EXP":
        shifts = config.shifts
        shifts_str = ", ".join(str(s) for s in shifts)
        p("Shifts", shifts_str)
    print("-" * len(header))


# ---------- Main RG drivers ---------- #


def qshe_rg_workflow(
    config: RGConfig,
    t_array: np.ndarray,
    f_array: np.ndarray,
    phi_array: np.ndarray,
    split_val: float,
    num_samples: int,
    output_index: int,
    matrix_batch_size: int,
    inputs: ArrayLike,
) -> np.ndarray:
    """Placeholder function for orchestrating the QSHE RG workflow. Currently not in use"""
    t_prime = numerical_solver(
        t_array,
        f_array,
        phi_array,
        split_val,
        num_samples,
        output_index,
        inputs,
        matrix_batch_size,
    )
    return t_prime


def rg_fp(
    rg_config: RGConfig,
    output_folders: dict,
    starting_t: int,
    starting_phi: int,
    starting_f: float = 0.0,
) -> dict:
    """Run an FP (fixed-point) RG workflow locally and write histograms.

    The function performs ``rg_config.steps`` iterations. For each step it
    computes the transformed samples using :func:`source.utilities.rg_data_workflow`,
    converts between t and z representations, computes histograms and writes
    NPZ files via :func:`source.utilities.save_data` into the folders provided
    by ``output_folders``.

    Parameters
    ----------
    rg_config : RGConfig
        Configuration dataclass containing numeric settings (samples, bins,
        ranges, resampling behaviour, seed, etc.).
    output_folders : dict
        Mapping with keys ``'t'`` and ``'z'`` giving output directories for
        t- and z-histograms respectively. Values should be string paths.
    starting_t : int
        If non-zero, indicates a fixed starting t value will be used.
    starting_phi : int
        If non-zero, indicates a fixed starting phi value will be used.

    Returns
    -------
    dict
        Mapping of step identifiers to the generated NPZ file paths, for
        example ``{"RG0": {"t": "...", "z": "..."}, ...}``.

    Notes
    -----
    - The implementation currently references an external ``args`` variable
      when constructing constant initial arrays if ``starting_t`` or
      ``starting_phi`` is non-zero. This variable is provided when the module
      is executed as a script; if you call :func:`rg_fp` programmatically you
      must supply ``starting_t``/``starting_phi`` values accordingly.
      (See module-level ``if __name__ == '__main__'`` block.)
    - Side effects: writes NPZ files to disk and prints progress to stdout.
    """
    samples = rg_config.samples
    batch_size = rg_config.matrix_batch_size
    steps = rg_config.steps
    # method = rg_config.method
    # expr = rg_config.expr
    resample = rg_config.resample
    symmetrise = rg_config.symmetrise
    seed = rg_config.seed
    t_bins = rg_config.t_bins
    t_range = rg_config.t_range
    z_bins = rg_config.z_bins
    z_range = rg_config.z_range
    inputs = rg_config.inputs
    rng = build_rng(seed)
    t_data_folder = output_folders["t"]
    z_data_folder = output_folders["z"]

    # Generate initial arrays
    initial_data = gen_initial_data(
        rg_config, starting_t, starting_phi, starting_f, rng
    )
    ts = initial_data["t"]
    fs = initial_data["f"]
    phases = initial_data["phi"]
    split = initial_data["split"]

    output_index = 2  # Track t' for now

    output_files = {}
    # Main rg loop
    for step in range(steps):
        print(f" Proceeding with RG step {step}. ")
        tprime = numerical_solver(
            ts, fs, phases, split, samples, output_index, inputs, batch_size
        )
        z = convert_t_to_z(tprime)
        t_data = build_hist(tprime, t_bins, t_range)
        z_data = build_hist(z, z_bins, z_range)
        if symmetrise == 1:
            print(" Symmetrising ")
            sym = "sym_"
            z_data["hist"] = center_z_distribution(z_data["hist"])
            z_sample = launder(
                samples,
                z_data["hist"],
                z_data["edges"],
                z_data["centers"],
                rng,
                resample,
            )
            t_sample = convert_z_to_t(z_sample)
        elif symmetrise == 0:
            sym = ""
            t_sample = launder(
                samples,
                t_data["hist"],
                t_data["edges"],
                t_data["centers"],
                rng,
                resample,
            )
        else:
            raise ValueError(f"Invalid symmetrise value entered: {symmetrise}")

        ts = extract_t_samples(t_sample, samples, rng)
        t_filename = f"{t_data_folder}/t_hist_RG{step + 1}.npz"
        z_filename = f"{z_data_folder}/z_{sym}hist_RG{step + 1}.npz"
        save_data(t_data["hist"], t_data["edges"], t_data["centers"], t_filename)
        save_data(z_data["hist"], z_data["edges"], z_data["centers"], z_filename)
        output_files.update({f"RG{step}": {"t": t_filename, "z": z_filename}})
    print(" All RG steps completed. ")
    return output_files


def rg_exp(
    rg_config: RGConfig, output_folders: dict, fp_dist: str, starting_phi: int
) -> dict:
    """Run an EXP (shifted / exponent) RG workflow locally and write histograms.

    Parameters
    ----------
    rg_config : RGConfig
        Configuration dataclass with samples, bins, ranges, shifts and other
        resampling parameters.
    output_folders : dict
        Mapping that, for each shift value, provides folders for ``t`` and
        ``z`` histograms (strings).
    fp_dist : str
        Path to a fixed-point NPZ file (containing keys ``'histval'``,
        ``'binedges'`` and ``'bincenters'``). The file is loaded to construct
        a laundered initial distribution.
    starting_phi : int
        If non-zero, a constant phase array is used; otherwise phases are
        generated randomly from RNG.

    Returns
    -------
    dict
        Nested mapping containing NPZ output paths per shift and RG step.

    Side effects
    ------------
    Writes NPZ files to disk (via :func:`source.utilities.save_data`) and prints
    progress to stdout.
    """
    samples = rg_config.samples
    batch_size = rg_config.matrix_batch_size
    steps = rg_config.steps
    method = rg_config.method
    expr = rg_config.expr
    resample = rg_config.resample
    symmetrise = rg_config.symmetrise
    seed = rg_config.seed
    t_bins = rg_config.t_bins
    t_range = rg_config.t_range
    z_bins = rg_config.z_bins
    z_range = rg_config.z_range
    shifts = [float(shift) for shift in rg_config.shifts]
    rng = build_rng(seed)
    if method == "analytic":
        i = 4
    else:
        i = 8
    output_files = {}
    fp_data = np.load(fp_dist)
    fp_hist = fp_data["histval"]
    fp_edges = fp_data["binedges"]
    fp_centers = fp_data["bincenters"]
    initial_z = launder(samples, fp_hist, fp_edges, fp_centers, rng, resample)
    for shift in shifts:
        t_data_folder = output_folders[f"{shift}"]["t"]
        z_data_folder = output_folders[f"{shift}"]["z"]
        shifted_z = initial_z + shift
        shifted_t = convert_z_to_t(shifted_z)
        if starting_phi != 0:
            phases = generate_constant_array(samples, starting_phi, i)
        else:
            phases = generate_random_phases(samples, rng, i)
        ts = extract_t_samples(shifted_t, samples, rng)
        for step in range(steps):
            print(f" Proceeding with RG step {step} of shift {shift}. ")
            tprime = rg_data_workflow(method, ts, phases, samples, expr, batch_size)
            z = convert_t_to_z(tprime)
            t_data = build_hist(tprime, t_bins, t_range)
            z_data = build_hist(z, z_bins, z_range)
            if symmetrise == 1:
                print(" Symmetrising ")
                sym = "sym_"
                z_data["hist"] = center_z_distribution(z_data["hist"])
                z_sample = launder(
                    samples,
                    z_data["hist"],
                    z_data["edges"],
                    z_data["centers"],
                    rng,
                    resample,
                )
                t_sample = convert_z_to_t(z_sample)
            elif symmetrise == 0:
                sym = ""
                t_sample = launder(
                    samples,
                    t_data["hist"],
                    t_data["edges"],
                    t_data["centers"],
                    rng,
                    resample,
                )
            else:
                raise ValueError(f"Invalid symmetrise value entered: {symmetrise}")
            ts = extract_t_samples(t_sample, samples, rng)
            t_filename = f"{t_data_folder}/t_hist_RG{step + 1}.npz"
            z_filename = f"{z_data_folder}/z_{sym}hist_RG{step + 1}.npz"
            save_data(t_data["hist"], t_data["edges"], t_data["centers"], t_filename)
            save_data(z_data["hist"], z_data["edges"], z_data["centers"], z_filename)
            output_files.update(
                {f"{shift}": {f"RG{step}": {"t": t_filename, "z": z_filename}}}
            )
        print(f" All RG steps of shift {shift} completed. ")
    return output_files


if __name__ == "__main__":
    cur_date = get_current_date()
    start_time = time()
    print(f" [{cur_date}]: Starting simulation.")
    # Build parser and read CLI args
    base_parser = build_parser()
    parser = append_parser(base_parser)
    parser.add_argument(
        "--type",
        required=True,
        default="FP",
        choices=["FP", "EXP"],
        help="Type of RG workflow",
    )
    args = parser.parse_args()
    args_dict = validate_input(args)

    # Process config
    config = handle_config(args_dict["config"], args.override)
    rg_config = build_config(config)

    # Make output folder and save config
    if args.out is None:
        base_output_dir = build_default_output_dir(config)
    else:
        base_output_dir = Path(args.out)
    output_dir = base_output_dir / args_dict["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    # Change stdout and stderr to other files for logging
    output_filename = f"{output_dir}/output.txt"
    error_filename = f"{output_dir}/error.txt"
    orig_output = sys.stdout
    orig_err = sys.stderr
    output_file = open(output_filename, "w")
    error_file = open(error_filename, "w")
    sys.stdout = output_file
    sys.stderr = error_file

    # Create children output folders for this workflow
    print_config(rg_config)
    output_folders = {}
    if args_dict["type"] == "EXP":
        shifts = [float(shift) for shift in rg_config.shifts]
        for shift in shifts:
            t_data_folder = output_dir / f"{shift}" / "hist/t"
            z_data_folder = output_dir / f"{shift}" / "hist/z"
            t_data_folder.mkdir(parents=True, exist_ok=True)
            z_data_folder.mkdir(parents=True, exist_ok=True)
            output_folders.update(
                {f"{shift}": {"t": str(t_data_folder), "z": str(z_data_folder)}}
            )
    else:
        t_data_folder = output_dir / "hist/t"
        z_data_folder = output_dir / "hist/z"
        t_data_folder.mkdir(parents=True, exist_ok=True)
        z_data_folder.mkdir(parents=True, exist_ok=True)
        output_folders.update({"t": str(t_data_folder), "z": str(z_data_folder)})

    print(f" Output folders: {json.dumps(output_folders, indent=2)} ")
    print("-" * 100)
    # Run RG workflow
    starting_t = args.t
    starting_phi = args.phi
    starting_f = args.f
    fp_data_file = f"{base_output_dir}/FP/hist/z/z_sym_hist_RG{rg_config.steps - 1}.npz"
    if args_dict["type"] == "FP":
        hist_outputs = rg_fp(
            rg_config, output_folders, starting_t, starting_phi, starting_f
        )
    else:
        hist_outputs = rg_exp(rg_config, output_folders, fp_data_file, starting_phi)
    print("-" * 100)

    # Closing off
    sys.stdout = orig_output
    sys.stderr = orig_err
    output_file.close()
    error_file.close()
    with open(f"{output_dir}/output_locs.json", "w") as file:
        json.dump(hist_outputs, file, indent=2)
    print(f" Outputs printed to {output_dir}. ")
    end_time = time()
    print(
        f" [{cur_date}]: Simulation completed after {end_time - start_time:.3f} seconds. "
    )
