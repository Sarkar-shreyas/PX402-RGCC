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
from typing import Optional
import numpy as np
import sys
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
    center_z_distribution,
    convert_zeff_to_t,
    get_current_date,
    build_rng,
    convert_z_to_g,
    convert_geff_to_t,
    convert_g_to_z,
    get_density,
    build_2d_hist,
    conditional_2d_resampler,
    convert_z_to_t,
    inverse_cdf_sampler,
)
from QSHE.testing_qshe import (
    # solve_qshe_matrix_eq,
    numerical_solver,
    gen_initial_data,
    append_parser,
)
from constants import THETA_DICT, local_dir

# ---------- Helper utility ---------- #


def build_default_output_dir(config: dict, theta_num: Optional[int] = None) -> Path:
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
    if theta_num is not None:
        return root / "Local data" / f"theta_{theta_num}" / version_str
    else:
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
    # if np.allclose(data, 0.0, 1e-12):
    #     # print("The data is really close to 0...")
    #     hist = 0.5 * (hist[::-1] + hist)
    centers = 0.5 * (edges[1:] + edges[:-1])
    densities = get_density(hist, edges)
    return {
        "histval": hist,
        "binedges": edges,
        "bincenters": centers,
        "densities": densities,
    }


def print_config(config: RGConfig, theta: int = 0) -> None:
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
    p("Initial mixing angle", THETA_DICT[str(theta)])
    if config.type.strip().upper() == "EXP":
        shifts = config.shifts
        shifts_str = ", ".join(str(s) for s in shifts)
        p("Shifts", shifts_str)
    print("-" * len(header))


# ---------- Main RG drivers ---------- #


def qshe_rg_workflow(
    config: RGConfig,
    output_dir: str | Path,
    vars: list,
    mode: str,
    eff: bool,
    starting_t: int,
    starting_phi: int,
    starting_th: int,
    gval: float = 0.0,
    fp_file: Optional[str] = None,
    two_dim: bool = True,
    y_var: Optional[str] = None,
    sample: bool = True,
) -> None:
    """Orchestrator for the QSHE RG workflow"""
    start = time()
    if two_dim:
        vars.append("2d")
    output_folders = create_output_folders(output_dir, vars, config)
    steps = config.steps
    samples = config.samples
    rng = build_rng(config.seed)

    outputs = config.outputs
    # inputs = config.inputs
    sym = bool(config.symmetrise)
    if mode == "FP":
        if fp_file is not None:
            initial_data = gen_initial_data(
                samples, starting_t, starting_phi, starting_th, rng, fp_file, gval=gval
            )
        else:
            initial_data = gen_initial_data(
                samples, starting_t, starting_phi, starting_th, rng, gval=gval
            )
        ts = initial_data["t"]
        fs = initial_data["f"]
        phases = initial_data["phi"]
        costheta = initial_data["theta"][0, 0]
        sintheta = np.sqrt(1 - costheta**2)
        print(f"Cos = {costheta}, Sin = {sintheta}")
        costheta_vals = []
        for step in range(steps):
            costheta_vals.append(costheta)
            data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)
            data_hists = construct_all_hists(config, data, two_dim, sym, y_var)
            for key, val in output_folders.items():
                save_hist(key, sym, val, step, data_hists[key])
            if sym:
                data_hists["z"]["histval"] = center_z_distribution(
                    data_hists["z"]["histval"]
                )
                z_sample = inverse_cdf_sampler(
                    samples,
                    data_hists["z"]["histval"],
                    data_hists["z"]["binedges"],
                    rng,
                )
            else:
                z_sample = data["z"]
            g_sample = convert_z_to_g(z_sample)
            t_sample = np.sqrt((costheta**2) * g_sample)
            f_sample = np.sqrt((sintheta**2) * g_sample)
            try:
                dist = np.abs(t_sample**2 + f_sample**2 - g_sample)
                assert np.allclose(dist, 1e-12)
            except AssertionError:
                print(
                    f"The distance from g is too large : Min = {np.min(dist)}, Max = {np.max(dist)}"
                )

            indexes = rng.integers(0, samples, size=(samples, 5))
            ts = np.take(t_sample, indexes)
            fs = np.take(f_sample, indexes)
            print(f"RG step {step} completed after {time() - start:.3f} seconds.")
        print(f"Costheta array : {costheta_vals}")
    else:
        shifts = config.shifts
        for shift in shifts:
            print(f"Proceeding with shift {shift}")
            print("-" * 100)
            initial_data = gen_initial_data(
                samples,
                starting_t,
                starting_phi,
                starting_th,
                rng,
                fp_file,
                shift,
                gval=gval,
            )
            ts = initial_data["t"]
            fs = initial_data["f"]
            phases = initial_data["phi"]
            costheta = initial_data["theta"][0, 0]
            sintheta = np.sqrt(1 - costheta**2)
            print(
                f"For shift {shift}, T mean = {np.mean(ts[0])}, F mean = {np.mean(fs[0])}"
            )
            print(f"Cos = {costheta}, Sin = {sintheta}")
            for step in range(steps):
                data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)
                data_hists = construct_all_hists(
                    config, data, two_dim, sym, y_var, shift
                )
                for key, val in output_folders[f"{shift}"].items():
                    save_hist(key, sym, val, step, data_hists[key])
                if sym:
                    data_hists["z"]["histval"] = center_z_distribution(
                        data_hists["z"]["histval"]
                    )
                    z_sample = inverse_cdf_sampler(
                        samples,
                        data_hists["z"]["histval"],
                        data_hists["z"]["binedges"],
                        rng,
                    )
                else:
                    z_sample = data["z"]

                g_sample = convert_z_to_g(z_sample)
                t_sample = np.sqrt((costheta**2) * g_sample)
                f_sample = np.sqrt((sintheta**2) * g_sample)
                try:
                    dist = np.abs(t_sample**2 + f_sample**2 - g_sample)
                    assert np.allclose(dist, 1e-12)
                except AssertionError:
                    print(
                        f"The distance from g is too large : Min = {np.min(dist)}, Max = {np.max(dist)}"
                    )
                indexes = rng.integers(0, samples, size=(samples, 5))
                ts = np.take(t_sample, indexes)
                fs = np.take(f_sample, indexes)
                print(
                    f"RG step {step} for shift {shift} completed after {time() - start:.3f} seconds."
                )
            print("-" * 100)


def single_qshe_rg_step(
    config: RGConfig,
    ts: np.ndarray,
    fs: np.ndarray,
    phis: np.ndarray,
    outputs: list,
    eff: bool,
) -> dict:
    """Computes the desired outputs from a single RG step for the input config and data"""
    n = config.samples
    batch_size = config.matrix_batch_size
    inputs = config.inputs
    index_var_map = {"2": "t", "9": "r", "10": "tau", "17": "f"}
    output_data = {}
    for index in outputs:
        data = numerical_solver(ts, fs, phis, n, index, inputs, batch_size)
        output_data.update({f"{index_var_map[f'{index}']}": data})

    tprime = output_data["t"]
    rprime = output_data["r"]
    tauprime = output_data["tau"]
    fprime = output_data["f"]
    try:
        output_sum = np.abs(tprime**2 + rprime**2 + tauprime**2 + fprime**2)
        abs_err = np.abs(output_sum - 1.0)
        assert np.all(abs_err < 1e-12)
    except AssertionError:
        print(
            f"The sum of outputs deviates from 1. Min : {np.min(abs_err)}, Max : {np.max(abs_err)}"
        )

    g = np.abs(tprime) ** 2 + np.abs(fprime) ** 2
    surv = np.abs(rprime) ** 2 + np.abs(tauprime) ** 2
    z = convert_g_to_z(g)
    mix = np.sqrt(tprime**2 / g)
    # assert np.allclose(mix, np.arcsin(np.sqrt(fprime**2 / g)), 1e-10)
    output_data.update({"mix": mix})
    output_data.update({"g": g, "z": z})
    output_data.update({"surv": surv})

    return output_data


def construct_all_hists(
    config: RGConfig,
    data_dict: dict,
    two_dim: bool,
    sym: bool,
    y_var: Optional[str] = None,
    shift: Optional[float] = None,
) -> dict:
    """Constructs all required histograms for the input data"""
    vars = data_dict.keys()
    t_bins = config.t_bins
    t_range = config.t_range
    z_bins = config.z_bins
    z_range = config.z_range
    if y_var is not None:
        second_var = y_var
    else:
        second_var = "mix"
    if shift is not None:
        z_range = (config.z_min + shift, config.z_max + shift)
    output_hists = {}
    for var in vars:
        data = data_dict[var]
        if var == "z":
            bins = z_bins
            range = z_range
        else:
            bins = t_bins
            range = t_range
        output_hists.update({var: build_hist(data, bins, range)})

    if two_dim:
        hist2d = build_2d_hist(
            ["z", second_var],
            data_dict["z"],
            data_dict[second_var],
            z_bins // 10,
            t_bins // 10,
            z_range,
            t_range,
            sym,
        )
        output_hists.update({"2d": hist2d})

    return output_hists


def create_output_folders(output_dir: str | Path, vars: list, config: RGConfig) -> dict:
    """Create all required folders for output vars"""
    rg_type = config.type
    if rg_type.lower() == "exp":
        shifts = config.shifts
        output_folders = {f"{shift}": {} for shift in shifts}
    else:
        shifts = None
        output_folders = {}

    if not isinstance(output_dir, Path):
        output_path = Path(output_dir)
    else:
        output_path = output_dir

    for var in vars:
        if shifts is not None:
            for shift in shifts:
                data_folder = output_path / "hist" / f"{shift}" / var
                data_folder.mkdir(parents=True, exist_ok=True)
                output_folders[f"{shift}"].update({var: str(data_folder)})
        else:
            data_folder = output_path / "hist" / var
            data_folder.mkdir(parents=True, exist_ok=True)
            output_folders.update({var: str(data_folder)})

    return output_folders


def save_hist(var: str, sym: bool, folder_name: str, rg_step: int, data: dict) -> None:
    if sym and var == "z":
        sym_text = "sym_"
    else:
        sym_text = ""
    filename = f"{folder_name}/{sym_text}{var}_hist_RG{rg_step}.npz"

    np.savez_compressed(filename, **data, allow_pickle=True)


def qshe_sampler(
    samples: int,
    rng: np.random.Generator,
    hist_dict: dict,
    y_var: Optional[str] = None,
) -> tuple:
    hist2d = hist_dict["2d"]
    if y_var is None:
        var2d = "z_mix"
    else:
        var2d = f"z_{y_var}"
    print("Starting sampling")
    sample1, sample2 = conditional_2d_resampler(hist2d, rng, samples, var2d)

    return sample1, sample2


if __name__ == "__main__":
    start_time = time()
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
    parser.add_argument("--g", type=float, default=0.5, help="Starting g value")
    parser.add_argument("--eff", action="store_true", help="Use the z_eff conversion?")
    parser.add_argument("--sample", action="store_true", help="Use resampler?")
    parser.add_argument(
        "--fpversion", default=None, help="Enter the version to get an FP from"
    )
    parser.add_argument(
        "--fpstep", default=None, help="Enter the RG step to use as an FP"
    )
    parser.add_argument(
        "--fpvar",
        default=None,
        help="Enter the name of the 2d var for the FP distribution",
    )
    args = parser.parse_args()
    args_dict = validate_input(args)

    # Process config
    config = handle_config(args_dict["config"], args.override)
    rg_config = build_config(config)
    rg_config.type = args_dict["type"]

    vars = rg_config.vars

    # Make output folder and save config
    if args.out is None:
        base_output_dir = build_default_output_dir(config, args.th)
    else:
        base_output_dir = Path(args.out)
    output_dir = base_output_dir / args_dict["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    cur_date = get_current_date()
    print(f"[{cur_date}]: Starting simulation.")
    # Change stdout and stderr to other files for logging
    output_filename = f"{output_dir}/output.txt"
    error_filename = f"{output_dir}/error.txt"
    print(f"Printing outputs to {output_filename} ")
    print(f"Printing errors to {error_filename}")
    orig_output = sys.stdout
    orig_err = sys.stderr
    output_file = open(output_filename, "w")
    error_file = open(error_filename, "w")
    sys.stdout = output_file
    sys.stderr = error_file

    # Create children output folders for this workflow
    print_config(rg_config, args.th)

    # print(f" Output folders: {json.dumps(output_folders, indent=2)} ")
    # print("-" * 100)
    # Run RG workflow
    starting_t = args.t
    starting_phi = args.phi
    starting_th = args.th
    # fp_data_file = f"{base_output_dir}/FP/hist/zf/zf_hist_RG7.npz"
    if args.fpversion is None or args.fpstep is None:
        if rg_config.type.lower() == "fp":
            fp_data_file = None
        else:
            fp_data_file = f"{base_output_dir}/FP/hist/{args.fpvar}/{args.fpvar}_hist_RG{rg_config.steps - 1}.npz"
    else:
        fp_data_file = f"{local_dir}/{args.fpversion}/FP/hist/{args.fpvar}/{args.fpvar}_hist_RG{args.fpstep}.npz"
    y_var = None
    two_dim = False
    qshe_rg_workflow(
        rg_config,
        output_dir,
        vars,
        rg_config.type,
        args.eff,
        starting_t,
        starting_phi,
        starting_th,
        args.g,
        fp_data_file,
        two_dim,
        y_var,
        args.sample,
    )

    # Closing off
    sys.stdout = orig_output
    sys.stderr = orig_err
    output_file.close()
    error_file.close()
    print(f"Outputs printed to {output_dir}. ")
    end_time = time()
    print(
        f"[{cur_date}]: Simulation completed after {end_time - start_time:.3f} seconds. "
    )
