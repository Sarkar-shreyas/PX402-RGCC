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
    convert_t_to_geff,
    convert_zeff_to_t,
    generate_constant_array,
    generate_random_phases,
    get_current_date,
    build_rng,
    convert_g_to_z,
    get_density,
    save_data,
    build_2d_hist,
    conditional_2d_resampler,
    convert_t_to_z,
    convert_t_to_g,
    convert_z_to_t,
)
from QSHE.testing_qshe import (
    # solve_qshe_matrix_eq,
    numerical_solver,
    gen_initial_data,
    append_parser,
)
from constants import local_dir

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
    return {
        "histval": hist,
        "binedges": edges,
        "bincenters": centers,
        "densities": densities,
    }


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
    output_dir: str | Path,
    vars: list,
    mode: str,
    eff: bool,
    starting_t: int,
    starting_phi: int,
    starting_f: float = 0.0,
    fp_file: Optional[str] = None,
    two_dim: bool = True,
    y_var: Optional[str] = None,
    sample: bool = True,
) -> None:
    """Placeholder function for orchestrating the QSHE RG workflow. Currently not in use"""
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
                samples, starting_t, starting_phi, starting_f, rng, fp_file
            )
        else:
            initial_data = gen_initial_data(
                samples, starting_t, starting_phi, starting_f, rng
            )
        ts = initial_data["t"]
        fs = initial_data["f"]
        phases = initial_data["phi"]
        for step in range(steps):
            data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)
            data_hists = construct_all_hists(config, data, two_dim, sym, y_var)
            # print(data_hists.keys())
            for key, val in output_folders.items():
                save_hist(key, sym, val, step, data_hists[key])
            if sample:
                z_sample, loss_sample = qshe_sampler(samples, rng, data_hists, y_var)
            else:
                z_sample = data["z"]
                loss_sample = data["leak"]
            if eff:
                t_sample = convert_zeff_to_t(z_sample, loss_sample)
            else:
                t_sample = convert_z_to_t(z_sample)
            indexes = rng.integers(0, samples, size=(samples, 5))
            ts = np.take(t_sample, indexes)
            fs = np.take(np.sqrt(loss_sample), indexes)
            print(f"RG step {step} completed after {time() - start:.3f} seconds.")
    else:
        shifts = config.shifts
        for shift in shifts:
            print(f"Proceeding with shift {shift}")
            print("-" * 100)
            initial_data = gen_initial_data(
                samples, starting_t, starting_phi, starting_f, rng, fp_file, shift
            )
            ts = initial_data["t"]
            fs = initial_data["f"]
            phases = initial_data["phi"]
            for step in range(steps):
                data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)
                data_hists = construct_all_hists(
                    config, data, two_dim, sym, y_var, shift
                )
                for key, val in output_folders[f"{shift}"].items():
                    save_hist(key, sym, val, step, data_hists[key])
                if sample:
                    z_sample, loss_sample = qshe_sampler(
                        samples, rng, data_hists, y_var
                    )
                else:
                    z_sample = data["z"]
                    loss_sample = data["leak"]
                if eff:
                    t_sample = convert_zeff_to_t(z_sample, loss_sample)
                else:
                    t_sample = convert_z_to_t(z_sample)
                indexes = rng.integers(0, samples, size=(samples, 5))
                ts = np.take(t_sample, indexes)
                fs = np.take(np.sqrt(loss_sample), indexes)
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

    if eff:
        g = convert_t_to_geff(tprime, rprime)
        z = convert_g_to_z(g)
    else:
        g = convert_t_to_g(tprime)
        z = convert_t_to_z(tprime)

    output_data.update({"g": g, "z": z})
    leak = (
        np.abs(np.reshape(tauprime, shape=n)) ** 2
        + np.abs(np.reshape(fprime, shape=n)) ** 2
    )
    surv = 1 - leak
    output_data.update({"leak": leak, "surv": surv})

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
        second_var = "f"
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
        var2d = "z_f"
    else:
        var2d = f"z_{y_var}"
    print("Starting sampling")
    sample1, sample2 = conditional_2d_resampler(hist2d, rng, samples, var2d)

    return sample1, sample2


# def rg_fp(
#     rg_config: RGConfig,
#     output_folders: dict,
#     starting_t: int,
#     starting_phi: int,
#     starting_f: float = 0.0,
# ) -> dict:
#     """Run an FP (fixed-point) RG workflow locally and write histograms.

#     The function performs ``rg_config.steps`` iterations. For each step it
#     computes the transformed samples using :func:`source.utilities.rg_data_workflow`,
#     converts between t and z representations, computes histograms and writes
#     NPZ files via :func:`source.utilities.save_data` into the folders provided
#     by ``output_folders``.

#     Parameters
#     ----------
#     rg_config : RGConfig
#         Configuration dataclass containing numeric settings (samples, bins,
#         ranges, resampling behaviour, seed, etc.).
#     output_folders : dict
#         Mapping with keys ``'t'`` and ``'z'`` giving output directories for
#         t- and z-histograms respectively. Values should be string paths.
#     starting_t : int
#         If non-zero, indicates a fixed starting t value will be used.
#     starting_phi : int
#         If non-zero, indicates a fixed starting phi value will be used.

#     Returns
#     -------
#     dict
#         Mapping of step identifiers to the generated NPZ file paths, for
#         example ``{"RG0": {"t": "...", "z": "..."}, ...}``.

#     Notes
#     -----
#     - The implementation currently references an external ``args`` variable
#       when constructing constant initial arrays if ``starting_t`` or
#       ``starting_phi`` is non-zero. This variable is provided when the module
#       is executed as a script; if you call :func:`rg_fp` programmatically you
#       must supply ``starting_t``/``starting_phi`` values accordingly.
#       (See module-level ``if __name__ == '__main__'`` block.)
#     - Side effects: writes NPZ files to disk and prints progress to stdout.
#     """
#     samples = rg_config.samples
#     batch_size = rg_config.matrix_batch_size
#     steps = rg_config.steps
#     # method = rg_config.method
#     # expr = rg_config.expr
#     # resample = rg_config.resample
#     symmetrise = rg_config.symmetrise
#     seed = rg_config.seed
#     t_bins = rg_config.t_bins
#     t_range = rg_config.t_range
#     z_bins = rg_config.z_bins
#     z_range = rg_config.z_range
#     inputs = rg_config.inputs
#     rng = build_rng(seed)
#     r_data_folder = output_folders["r"]
#     t_data_folder = output_folders["t"]
#     tau_data_folder = output_folders["tau"]
#     f_data_folder = output_folders["f"]
#     loss_data_folder = output_folders["loss"]
#     z_data_folder = output_folders["z"]
#     zf_data_folder = output_folders["zf"]
#     # Generate initial arrays
#     initial_data = gen_initial_data(
#         rg_config.samples, starting_t, starting_phi, starting_f, rng
#     )
#     ts = initial_data["t"]
#     fs = initial_data["f"]
#     phases = initial_data["phi"]
#     z_2d_bins = z_bins // 100
#     t_2d_bins = t_bins // 10
#     # output_index = 9  # Track t'

#     output_files = {}
#     # Main rg loop
#     for step in range(steps):
#         print(f"Proceeding with RG step {step}. ")
#         # For now, track all 4 outputs.
#         tprime = numerical_solver(ts, fs, phases, samples, 2, inputs, batch_size)
#         rprime = numerical_solver(ts, fs, phases, samples, 9, inputs, batch_size)
#         tauprime = numerical_solver(ts, fs, phases, samples, 10, inputs, batch_size)
#         fprime = numerical_solver(ts, fs, phases, samples, 17, inputs, batch_size)
#         try:
#             output_sum = np.abs(tprime**2 + rprime**2 + tauprime**2 + fprime**2)
#             abs_err = np.abs(output_sum - 1.0)
#             assert np.all(abs_err < 1e-12)
#         except AssertionError:
#             print(
#                 f"The sum of outputs deviates from 1. Min : {np.min(abs_err)}, Max : {np.max(abs_err)}"
#             )
#         g_eff = convert_t_to_geff(tprime, rprime)
#         z = convert_g_to_z(g_eff)
#         loss = (
#             np.abs(np.reshape(tauprime, shape=samples)) ** 2
#             + np.abs(np.reshape(fprime, shape=samples)) ** 2
#         )
#         # z = convert_t_to_z(tprime)
#         t_data = build_hist(tprime, t_bins, t_range)
#         r_data = build_hist(rprime, t_bins, t_range)
#         tau_data = build_hist(tauprime, t_bins, t_range)
#         f_data = build_hist(fprime, t_bins, t_range)
#         loss_data = build_hist(loss, t_bins, t_range)
#         z_data = build_hist(z, z_bins, z_range)
#         print(f"Is 0.0 a bin edge? {np.any(np.isclose(z_data['edges'], 0.0))}")
#         if symmetrise == 1:
#             print("Symmetrising")
#             hist_dict = build_2d_hist(
#                 z, loss, z_2d_bins, t_2d_bins, z_range, t_range, True
#             )
#             sym_z_filename = f"{z_data_folder}/sym_z_hist_RG{step}.npz"
#             save_data(
#                 hist_dict["z"]["counts"],
#                 hist_dict["z"]["binedges"],
#                 hist_dict["z"]["bincenters"],
#                 sym_z_filename,
#             )
#             print(
#                 f"Is 0.0 a bin edge of z post sym? {np.any(np.isclose(hist_dict['z']['binedges'], 0.0))}"
#             )
#             print(
#                 f"Sum of reflected centers: {np.max(np.abs(hist_dict['z']['bincenters'] + hist_dict['z']['bincenters'][::-1]))}"
#             )
#             z_sample, loss_sample = conditional_2d_resampler(hist_dict, rng, samples)
#             t_sample = convert_zeff_to_t(z_sample, loss_sample)
#         elif symmetrise == 0:
#             hist_dict = build_2d_hist(z, loss, z_2d_bins, t_2d_bins, z_range, t_range)
#             z_sample, loss_sample = conditional_2d_resampler(hist_dict, rng, samples)
#             t_sample = convert_zeff_to_t(z_sample, loss_sample)
#         else:
#             raise ValueError(f"Invalid symmetrise value entered: {symmetrise}")

#         indexes = rng.integers(0, samples, size=(samples, 5))
#         ts = np.take(t_sample, indexes)
#         fs = np.take(np.sqrt(loss_sample), indexes)
#         assert ts.shape == (samples, 5) and fs.shape == (samples, 5)
#         t_filename = f"{t_data_folder}/t_hist_RG{step}.npz"
#         r_filename = f"{r_data_folder}/r_hist_RG{step}.npz"
#         tau_filename = f"{tau_data_folder}/tau_hist_RG{step}.npz"
#         f_filename = f"{f_data_folder}/f_hist_RG{step}.npz"
#         loss_filename = f"{loss_data_folder}/loss_hist_RG{step}.npz"
#         z_filename = f"{z_data_folder}/z_hist_RG{step}.npz"
#         zf_filename = f"{zf_data_folder}/zf_hist_RG{step}.npz"
#         save_data(t_data["hist"], t_data["edges"], t_data["centers"], t_filename)
#         save_data(r_data["hist"], r_data["edges"], r_data["centers"], r_filename)
#         save_data(
#             tau_data["hist"], tau_data["edges"], tau_data["centers"], tau_filename
#         )
#         save_data(f_data["hist"], f_data["edges"], f_data["centers"], f_filename)
#         save_data(
#             loss_data["hist"], loss_data["edges"], loss_data["centers"], loss_filename
#         )
#         save_data(z_data["hist"], z_data["edges"], z_data["centers"], z_filename)
#         np.savez_compressed(
#             zf_filename,
#             zfcounts=hist_dict["zf"]["counts"],
#             zfdensities=hist_dict["zf"]["densities"],
#             zcounts=hist_dict["z"]["counts"],
#             zbins=hist_dict["z"]["binedges"],
#             zcenters=hist_dict["z"]["bincenters"],
#             zdensities=hist_dict["z"]["densities"],
#             fcounts=hist_dict["f"]["counts"],
#             fbins=hist_dict["f"]["binedges"],
#             fcenters=hist_dict["f"]["bincenters"],
#             fdensities=hist_dict["f"]["densities"],
#         )
#         output_files.update(
#             {
#                 f"RG{step}": {
#                     "t": t_filename,
#                     "r": r_filename,
#                     "tau": tau_filename,
#                     "f": f_filename,
#                     "loss": loss_filename,
#                     "z": z_filename,
#                     "zf": zf_filename,
#                 }
#             }
#         )
#     print(" All RG steps completed. ")
#     return output_files


# def rg_exp(
#     rg_config: RGConfig, output_folders: dict, fp_dist: str, starting_phi: int
# ) -> dict:
#     """Run an EXP (shifted / exponent) RG workflow locally and write histograms.

#     Parameters
#     ----------
#     rg_config : RGConfig
#         Configuration dataclass with samples, bins, ranges, shifts and other
#         resampling parameters.
#     output_folders : dict
#         Mapping that, for each shift value, provides folders for ``t`` and
#         ``z`` histograms (strings).
#     fp_dist : str
#         Path to a fixed-point NPZ file (containing keys ``'histval'``,
#         ``'binedges'`` and ``'bincenters'``). The file is loaded to construct
#         a laundered initial distribution.
#     starting_phi : int
#         If non-zero, a constant phase array is used; otherwise phases are
#         generated randomly from RNG.

#     Returns
#     -------
#     dict
#         Nested mapping containing NPZ output paths per shift and RG step.

#     Side effects
#     ------------
#     Writes NPZ files to disk (via :func:`source.utilities.save_data`) and prints
#     progress to stdout.
#     """
#     samples = rg_config.samples
#     batch_size = rg_config.matrix_batch_size
#     steps = rg_config.steps
#     # method = rg_config.method
#     # expr = rg_config.expr
#     # resample = rg_config.resample
#     symmetrise = rg_config.symmetrise
#     seed = rg_config.seed
#     t_bins = rg_config.t_bins
#     t_range = rg_config.t_range
#     z_bins = rg_config.z_bins
#     z_range = rg_config.z_range
#     inputs = rg_config.inputs
#     shifts = [float(shift) for shift in rg_config.shifts]
#     rng = build_rng(seed)
#     i = 16
#     # Load FP data
#     fp_data = np.load(fp_dist)
#     fp_zf_counts = fp_data["zfcounts"]
#     fp_z_binedges = fp_data["zbins"]
#     fp_loss_binedges = fp_data["fbins"]
#     fp_dict = {
#         "zf": {"counts": fp_zf_counts},
#         "z": {"binedges": fp_z_binedges},
#         "f": {"binedges": fp_loss_binedges},
#     }
#     z_sample, loss_sample = conditional_2d_resampler(fp_dict, rng, samples)
#     output_files = {}
#     initial_z = z_sample
#     z_2d_bins = z_bins // 100
#     t_2d_bins = t_bins // 10
#     for shift in shifts:
#         r_data_folder = output_folders[f"{shift}"]["r"]
#         t_data_folder = output_folders[f"{shift}"]["t"]
#         tau_data_folder = output_folders[f"{shift}"]["tau"]
#         f_data_folder = output_folders[f"{shift}"]["f"]
#         loss_data_folder = output_folders[f"{shift}"]["loss"]
#         z_data_folder = output_folders[f"{shift}"]["z"]
#         zf_data_folder = output_folders[f"{shift}"]["zf"]
#         shifted_z = initial_z + shift
#         z_min, z_max = z_range
#         shifted_zmin = z_min + shift
#         shifted_zmax = z_max + shift
#         shifted_zrange = (shifted_zmin, shifted_zmax)
#         shifted_t = convert_zeff_to_t(shifted_z, loss_sample)
#         if starting_phi != 0:
#             phases = generate_constant_array(samples, starting_phi, i)
#         else:
#             phases = generate_random_phases(samples, rng, i)
#         indexes = rng.integers(0, samples, size=(samples, 5))
#         ts = np.take(shifted_t, indexes)
#         fs = np.take(np.sqrt(loss_sample), indexes)
#         assert ts.shape == (samples, 5) and fs.shape == (samples, 5)
#         for step in range(steps):
#             print(f" Proceeding with RG step {step} of shift {shift}. ")
#             tprime = numerical_solver(ts, fs, phases, samples, 2, inputs, batch_size)
#             rprime = numerical_solver(ts, fs, phases, samples, 9, inputs, batch_size)
#             tauprime = numerical_solver(ts, fs, phases, samples, 10, inputs, batch_size)
#             fprime = numerical_solver(ts, fs, phases, samples, 17, inputs, batch_size)
#             try:
#                 output_sum = np.abs(tprime**2 + rprime**2 + tauprime**2 + fprime**2)
#                 abs_err = np.abs(output_sum - 1.0)
#                 assert np.all(abs_err < 1e-12)
#             except AssertionError:
#                 print(
#                     f"The sum of outputs deviates from 1. Min : {np.min(abs_err)}, Max : {np.max(abs_err)}"
#                 )
#             g_eff = convert_t_to_geff(tprime, rprime)
#             z = convert_g_to_z(g_eff)
#             loss = (
#                 np.abs(np.reshape(tauprime, shape=samples)) ** 2
#                 + np.abs(np.reshape(fprime, shape=samples)) ** 2
#             )
#             t_data = build_hist(tprime, t_bins, t_range)
#             r_data = build_hist(rprime, t_bins, t_range)
#             tau_data = build_hist(tauprime, t_bins, t_range)
#             f_data = build_hist(fprime, t_bins, t_range)
#             loss_data = build_hist(loss, t_bins, t_range)
#             z_data = build_hist(z, z_bins, shifted_zrange)
#             if symmetrise == 1:
#                 print("Symmetrising")
#                 hist_dict = build_2d_hist(
#                     z, loss, z_2d_bins, t_2d_bins, shifted_zrange, t_range, True
#                 )
#                 sym_z_filename = f"{z_data_folder}/sym_z_hist_RG{step}.npz"
#                 save_data(
#                     hist_dict["z"]["counts"],
#                     hist_dict["z"]["binedges"],
#                     hist_dict["z"]["bincenters"],
#                     sym_z_filename,
#                 )
#                 z_sample, loss_sample = conditional_2d_resampler(
#                     hist_dict, rng, samples
#                 )
#                 shifted_t = convert_zeff_to_t(z_sample, loss_sample)
#             elif symmetrise == 0:
#                 hist_dict = build_2d_hist(
#                     z, loss, z_2d_bins, t_2d_bins, shifted_zrange, t_range
#                 )
#                 z_sample, loss_sample = conditional_2d_resampler(
#                     hist_dict, rng, samples
#                 )
#                 shifted_t = convert_zeff_to_t(z_sample, loss_sample)
#             else:
#                 raise ValueError(f"Invalid symmetrise value entered: {symmetrise}")
#             indexes = rng.integers(0, samples, size=(samples, 5))
#             ts = np.take(shifted_t, indexes)
#             fs = np.take(np.sqrt(loss_sample), indexes)
#             t_filename = f"{t_data_folder}/t_hist_RG{step}.npz"
#             r_filename = f"{r_data_folder}/r_hist_RG{step}.npz"
#             tau_filename = f"{tau_data_folder}/tau_hist_RG{step}.npz"
#             f_filename = f"{f_data_folder}/f_hist_RG{step}.npz"
#             loss_filename = f"{loss_data_folder}/loss_hist_RG{step}.npz"
#             z_filename = f"{z_data_folder}/z_hist_RG{step}.npz"
#             zf_filename = f"{zf_data_folder}/zf_hist_RG{step}.npz"
#             save_data(t_data["hist"], t_data["edges"], t_data["centers"], t_filename)
#             save_data(r_data["hist"], r_data["edges"], r_data["centers"], r_filename)
#             save_data(
#                 tau_data["hist"], tau_data["edges"], tau_data["centers"], tau_filename
#             )
#             save_data(f_data["hist"], f_data["edges"], f_data["centers"], f_filename)
#             save_data(
#                 loss_data["hist"],
#                 loss_data["edges"],
#                 loss_data["centers"],
#                 loss_filename,
#             )
#             save_data(z_data["hist"], z_data["edges"], z_data["centers"], z_filename)
#             np.savez_compressed(
#                 zf_filename,
#                 zfcounts=hist_dict["zf"]["counts"],
#                 zfdensities=hist_dict["zf"]["densities"],
#                 zcounts=hist_dict["z"]["counts"],
#                 zbins=hist_dict["z"]["binedges"],
#                 zcenters=hist_dict["z"]["bincenters"],
#                 zdensities=hist_dict["z"]["densities"],
#                 fcounts=hist_dict["f"]["counts"],
#                 fbins=hist_dict["f"]["binedges"],
#                 fcenters=hist_dict["f"]["bincenters"],
#                 fdensities=hist_dict["f"]["densities"],
#             )
#             output_files.update(
#                 {
#                     f"{shift}": {
#                         f"RG{step}": {
#                             "t": t_filename,
#                             "r": r_filename,
#                             "tau": tau_filename,
#                             "f": f_filename,
#                             "loss": loss_filename,
#                             "z": z_filename,
#                             "zf": zf_filename,
#                         }
#                     }
#                 }
#             )
#         print(f" All RG steps of shift {shift} completed. ")
#     return output_files


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
        base_output_dir = build_default_output_dir(config)
    else:
        base_output_dir = Path(args.out)
    output_dir = base_output_dir / args_dict["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    cur_date = get_current_date()
    print(f" [{cur_date}]: Starting simulation.")
    # Change stdout and stderr to other files for logging
    output_filename = f"{output_dir}/output.txt"
    error_filename = f"{output_dir}/error.txt"
    orig_output = sys.stdout
    orig_err = sys.stderr
    # output_file = open(output_filename, "w")
    # error_file = open(error_filename, "w")
    # sys.stdout = output_file
    # sys.stderr = error_file

    # Create children output folders for this workflow
    print_config(rg_config)

    # print(f" Output folders: {json.dumps(output_folders, indent=2)} ")
    # print("-" * 100)
    # Run RG workflow
    starting_t = args.t
    starting_phi = args.phi
    starting_f = args.f
    # fp_data_file = f"{base_output_dir}/FP/hist/zf/zf_hist_RG7.npz"
    if args.fpversion is None or args.fpstep is None:
        if rg_config.type == "fp":
            fp_data_file = None
        else:
            fp_data_file = f"{base_output_dir}/FP/hist/{args.fpvar}/{args.fpvar}_hist_RG{rg_config.steps - 1}.npz"
    else:
        fp_data_file = f"{local_dir}/{args.fpversion}/FP/hist/{args.fpvar}/{args.fpvar}_hist_RG{args.fpstep}.npz"
    y_var = None
    two_dim = True
    qshe_rg_workflow(
        rg_config,
        output_dir,
        vars,
        rg_config.type,
        args.eff,
        starting_t,
        starting_phi,
        starting_f,
        fp_data_file,
        two_dim,
        y_var,
        args.sample,
    )

    # Closing off
    # sys.stdout = orig_output
    # sys.stderr = orig_err
    # output_file.close()
    # error_file.close()
    print(f"Outputs printed to {output_dir}. ")
    end_time = time()
    print(
        f"[{cur_date}]: Simulation completed after {end_time - start_time:.3f} seconds. "
    )
