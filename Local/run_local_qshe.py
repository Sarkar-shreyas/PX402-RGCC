"""Local QSHE RG driver for development and testing.

Purpose
-------
Single-process driver for running QSHE Renormalization Group (RG) workflows
locally, intended for development and testing only — not for production use.
Re-uses the same ``source/`` and ``QSHE/`` library code as the Taskfarm HPC
scripts but runs in one Python process without Slurm job arrays.

Unlike the IQHE pipeline, the QSHE model tracks two coupled amplitudes:
the transmission amplitude ``t`` and the forward-scattering amplitude ``f``.
The mixing angle θ (``--q``) governs the initial balance between them.

Differences from HPC Taskfarm scripts
--------------------------------------
- **No Slurm**: all computation is sequential in one process; HPC uses
  Slurm job arrays via ``Taskfarm/scripts/``.
- **No aggregation step**: the Taskfarm pipeline writes per-q-block ``.npy``
  files later combined by ``source/qshe_data_agg.py``; this driver handles
  the full workflow inline.

CLI Usage
---------
Run from the repository root::

    python -m Local.run_local_qshe \\
        --config Local/configs/local_qshe \\
        --type FP \\
        --q 0.0

    python -m Local.run_local_qshe \\
        --config Local/configs/local_qshe \\
        --type EXP \\
        --q 0.0 \\
        --fpvar z \\
        --fpstep 6

Output Location
---------------
Outputs are written under ``Local data/`` in the repository root::

    Local data/q_{theta_num}/{version}_{method}_{expr}/FP/
        hist/{var}/{sym_}{var}_hist_RG{i}.npz
        output.txt
        error.txt
        updated_config.yaml

    Local data/q_{theta_num}/{version}_{method}_{expr}/EXP/
        hist/{shift}/{var}/{sym_}{var}_hist_RG{i}.npz

Notes
-----
- stdout and stderr are redirected to ``output.txt`` and ``error.txt``
  inside the output directory when the module is executed as a script.
- QSHE analysis (post-processing, visualisation, ν extraction) is performed
  separately in ``test_qshe.ipynb`` — there is no CLI analysis equivalent.
- The EXP workflow seeds its initial distribution from a previously run FP
  histogram; run FP before EXP for a given config.
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
    """Build the default local output directory for a QSHE config.

    Args:
        config: Parsed configuration dictionary (result of
            :func:`source.parse_config.validate_input` /
            :func:`source.config.handle_config`). Must include
            ``main.version``, ``engine.method``, and ``engine.expr`` keys.
        theta_num: Optional mixing-angle index (the ``--q`` CLI argument).
            When provided, an extra ``q_{theta_num}`` level is inserted so
            that runs at different θ values do not share an output path.

    Returns:
        A path under the repository root:

        - ``<repo_root>/Local data/q_{theta_num}/{version}_{method}_{expr}``
          when *theta_num* is given.
        - ``<repo_root>/Local data/{version}_{method}_{expr}`` otherwise.
    """
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root(1)
    if theta_num is not None:
        return root / "Local data" / f"q_{theta_num}" / version_str
    else:
        return root / "Local data" / version_str


def build_hist(data: np.ndarray, bins: int, range: tuple) -> dict:
    """Compute a histogram and return related arrays and densities.

    Args:
        data: 1-D array of samples to histogram.
        bins: Number of histogram bins.
        range: ``(min, max)`` binning range.

    Returns:
        Dictionary with keys: ``histval`` (counts), ``binedges`` (bin
        edges), ``bincenters`` (bin centers) and ``densities`` (density
        per bin computed using :func:`source.utilities.get_density`).
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


def print_config(config: RGConfig, theta: float = 0.0) -> None:
    """Print a compact, human-readable summary of the main run settings.

    Args:
        config: Configuration dataclass returned by
            :func:`source.config.build_config`.
        theta: Initial QSHE mixing angle θ (the ``--q`` CLI argument).
            Printed alongside the other run settings for reference.

    Notes:
        If the configuration indicates an ``EXP`` run, this function expects
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
    p("Initial mixing angle", theta)
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
    """Orchestrate the full QSHE RG workflow for FP or EXP mode.

    Dispatches to the FP (fixed-point) or EXP (shifted) branch depending on
    *mode*, creates per-variable output folders, then runs ``config.steps``
    RG iterations calling :func:`single_qshe_rg_step` and
    :func:`construct_all_hists` at each step.

    FP branch sequence:

    1. Draw an initial ``(t, f, phi)`` sample via
       :func:`QSHE.testing_qshe.gen_initial_data` (optionally seeded from
       *fp_file*).
    2. For each step: apply the RG map, build histograms for every variable,
       optionally symmetrise the z-distribution, then resample ``(t, f)``
       for the next step.

    EXP branch sequence:

    1. For each shift in ``config.shifts``: construct a perturbed initial
       sample from *fp_file* via :func:`QSHE.testing_qshe.gen_initial_data`
       with the shift applied.
    2. Run the same per-step sequence as the FP branch.

    Args:
        config: Validated :class:`~source.config.RGConfig` dataclass for
            the run.
        output_dir: Root output directory.  Per-variable and per-shift
            sub-folders are created inside it by
            :func:`create_output_folders`.
        vars: Observable names to track (e.g. ``["z", "t", "f", "g"]``).
            If *two_dim* is ``True``, ``"2d"`` is appended automatically.
        mode: ``"FP"`` for fixed-point run, ``"EXP"`` for shifted run.
        eff: When ``True``, uses the effective-z conversion in downstream
            computations.
        starting_t: Starting t-amplitude selector passed to
            :func:`QSHE.testing_qshe.gen_initial_data`.
        starting_phi: Starting phase selector passed to
            :func:`QSHE.testing_qshe.gen_initial_data`.
        starting_th: Starting mixing-angle index (maps to θ) passed to
            :func:`QSHE.testing_qshe.gen_initial_data`.
        gval: Initial conductance g value passed to
            :func:`QSHE.testing_qshe.gen_initial_data`. Default: ``0.0``.
        fp_file: Optional path to a fixed-point histogram ``.npz`` file
            used to seed the initial distribution for FP continuation and
            EXP runs.  ``None`` means a fresh random draw.
        two_dim: When ``True``, a 2-D (z, *y_var*) joint histogram is
            built at each step and stored under the ``"2d"`` key.
            Default: ``True``.
        y_var: Name of the second variable for the 2-D histogram.
            Defaults to ``"mix"`` when ``None``.
        sample: Reserved flag passed through the CLI (currently unused
            inside this function but preserved for API compatibility).

    Side effects:
        Creates sub-directories, writes histogram NPZ files to disk, and
        prints step-level progress to stdout.
    """
    start = time()
    # Append the "2d" key so create_output_folders makes the joint-histogram folder
    if two_dim:
        vars.append("2d")
    output_folders = create_output_folders(output_dir, vars, config)
    steps = config.steps
    samples = config.samples
    rng = build_rng(config.seed)

    outputs = config.outputs
    # inputs = config.inputs
    sym = bool(config.symmetrise)

    # --- FP branch: single fixed-point run ---
    if mode == "FP":
        # Generate the initial (t, f, phi) sample; use fp_file when continuing from a prior run
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
        q = starting_th
        print(f"Performing RG workflow for q = {q}")

        # --- FP RG iteration loop ---
        for step in range(steps):
            # Apply the RG map to produce next-step observables
            data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)

            # Build histograms for all tracked variables
            data_hists = construct_all_hists(config, data, two_dim, sym, y_var)

            # Write histogram NPZ files for each variable at this step
            for key, val in output_folders.items():
                save_hist(key, sym, val, step, data_hists[key])

            # Symmetrisation branch: fold z-distribution then resample via inverse CDF.
            # Unsymmetrised branch: derive t and f directly from the step output.
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
                p_sample = data["p"]
                f_sample = data["f"]
                t_sample = np.sqrt(p_sample)
                q_dist = (f_sample**2) / (1 - p_sample)
                print(
                    f"Stats of q after RG step {step}; Mean : {np.mean(q_dist):.3f}, Median : {np.median(q_dist):.3f}, Min : {np.min(q_dist):.7f}, Max : {np.max(q_dist):.3f}"
                )

            # Resample (t, f) pairs from the step output for the next iteration
            indexes = rng.integers(0, samples, size=(samples, 5))
            ts = np.take(t_sample, indexes)
            fs = np.take(f_sample, indexes)
            print(f"RG step {step} completed after {time() - start:.3f} seconds.")

    # --- EXP branch: shifted runs for each perturbation magnitude ---
    else:
        shifts = config.shifts
        for shift in shifts:
            print(f"Proceeding with shift {shift}")
            print("-" * 100)

            # Build the perturbed initial sample from the FP distribution
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

            # --- EXP RG iteration loop for this shift ---
            for step in range(steps):
                # Apply the RG map to produce next-step observables
                data = single_qshe_rg_step(config, ts, fs, phases, outputs, eff)

                # Build histograms for all tracked variables (z-range shifted for EXP)
                data_hists = construct_all_hists(
                    config, data, two_dim, sym, y_var, shift
                )

                # Write histogram NPZ files for this shift and step
                for key, val in output_folders[f"{shift}"].items():
                    save_hist(key, sym, val, step, data_hists[key])

                # Symmetrisation branch or direct z→g→(t,f) conversion
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

                # Recover (t, f) from conductance g, preserving the θ-split
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

                # Resample (t, f) pairs for the next iteration
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
    """Compute QSHE observables for a single RG step.

    Calls :func:`QSHE.testing_qshe.numerical_solver` for each requested
    output index, then derives secondary observables (g, p, z, surv, mix)
    from the primary amplitudes and validates unitarity.

    Args:
        config: Validated :class:`~source.config.RGConfig` dataclass for
            the run. Used for ``samples``, ``matrix_batch_size``, and
            ``inputs``.
        ts: Transmission amplitude array of shape ``(samples, 5)``.
        fs: Forward-scattering amplitude array of shape ``(samples, 5)``.
        phis: Phase array of shape ``(samples, n_phases)``.
        outputs: List of integer output-index codes selecting which
            primary amplitudes to compute (e.g. ``[2, 9, 10, 17]`` for
            t, r, τ, f).
        eff: When ``True``, uses the effective-z conversion (passed
            through but does not alter the outputs computed here).

    Returns:
        Dictionary of arrays for all computed observables:
        ``t``, ``r``, ``tau``, ``f`` (primary amplitudes from the solver),
        ``g`` (conductance = t² + f²),
        ``p`` (transmission probability = t²),
        ``surv`` (survival probability = |r|² + |τ|²),
        ``z`` (log-ratio via :func:`source.utilities.convert_g_to_z`),
        ``mix`` (mixing ratio = f²/(1−p)).

    Raises:
        AssertionError: If the unitarity check |t² + r² + τ² + f²| ≈ 1
            fails, or if g is inconsistent with the inverse z→g conversion.
    """
    n = config.samples
    batch_size = config.matrix_batch_size
    inputs = config.inputs
    # Map numerical solver output indices to human-readable variable names
    index_var_map = {"2": "t", "9": "r", "10": "tau", "17": "f"}
    output_data = {}

    # --- Compute each requested primary amplitude via the numerical solver ---
    for index in outputs:
        data = numerical_solver(ts, fs, phis, n, index, inputs, batch_size)
        output_data.update({f"{index_var_map[f'{index}']}": data})

    tprime = output_data["t"]
    rprime = output_data["r"]
    tauprime = output_data["tau"]
    fprime = output_data["f"]

    # --- Unitarity check: |t'² + r'² + τ'² + f'²| must be ≈ 1 ---
    try:
        output_sum = np.abs(tprime**2 + rprime**2 + tauprime**2 + fprime**2)
        abs_err = np.abs(output_sum - 1.0)
        assert np.all(abs_err < 1e-12)
    except AssertionError:
        print(
            f"The sum of outputs deviates from 1. Min : {np.min(abs_err)}, Max : {np.max(abs_err)}"
        )

    # --- Derive secondary observables from primary amplitudes ---
    g = tprime**2 + fprime**2
    p = tprime**2
    surv = np.abs(rprime) ** 2 + np.abs(tauprime) ** 2
    z = convert_g_to_z(g)
    # z = np.log(surv / g)
    # mix = np.sqrt(surv)
    mix = (fprime**2) / (1 - p)
    # assert np.allclose(mix, np.arcsin(np.sqrt(fprime**2 / g)), 1e-10)
    assert np.allclose(g, convert_z_to_g(z), 1e-10)
    assert np.allclose(g + surv, 1.0, 1e-10)
    output_data.update({"p": p})
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
    """Build histograms for all observables in *data_dict*.

    Args:
        config: Validated :class:`~source.config.RGConfig` dataclass.
            Provides ``t_bins``, ``t_range``, ``z_bins``, ``z_range``,
            ``z_min``, and ``z_max``.
        data_dict: Mapping of observable name → 1-D sample array.
            Keys determine which variables are histogrammed.
        two_dim: When ``True``, also builds a 2-D joint histogram of
            (z, *y_var*) via :func:`source.utilities.build_2d_hist` and
            stores it under the ``"2d"`` key.
        sym: Passed to :func:`source.utilities.build_2d_hist`; controls
            whether the z-axis of the 2-D histogram is symmetrised.
        y_var: Name of the second variable for the 2-D histogram.
            Defaults to ``"mix"`` when ``None``.
        shift: Optional float perturbation applied to ``z_range``
            boundaries (used for EXP runs so the z-histogram spans the
            shifted distribution rather than the fixed-point one).

    Returns:
        Dictionary mapping each observable name (plus ``"2d"`` when
        *two_dim* is ``True``) to its histogram dict (as returned by
        :func:`build_hist` or :func:`source.utilities.build_2d_hist`).
    """
    vars = data_dict.keys()
    t_bins = config.t_bins
    t_range = config.t_range
    z_bins = config.z_bins
    z_range = config.z_range
    if y_var is not None:
        second_var = y_var
    else:
        second_var = "mix"
    # For EXP runs, translate the z-range boundaries by the shift so the
    # histogram spans the perturbed distribution rather than the FP distribution
    if shift is not None:
        z_range = (config.z_min + shift, config.z_max + shift)
    output_hists = {}

    # --- Build a 1-D histogram for each tracked observable ---
    for var in vars:
        data = data_dict[var]
        if var == "z":
            bins = z_bins
            range = z_range
        else:
            bins = t_bins
            range = t_range
        output_hists.update({var: build_hist(data, bins, range)})

    # --- Optionally build the 2-D joint (z, y_var) histogram ---
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
    """Create per-variable (and per-shift for EXP) histogram output folders.

    For FP runs the folder structure is::

        {output_dir}/hist/{var}/

    For EXP runs the folder structure is::

        {output_dir}/hist/{shift}/{var}/

    Args:
        output_dir: Root output directory path (str or
            :class:`pathlib.Path`).
        vars: List of observable variable names (e.g.
            ``["z", "t", "2d"]``).  One sub-directory is created per
            variable.
        config: Validated :class:`~source.config.RGConfig` dataclass.
            Used to read ``type`` and ``shifts`` for the EXP folder layout.

    Returns:
        For FP runs — a flat dict mapping variable name → folder path
        string::

            {"z": ".../hist/z", "t": ".../hist/t", ...}

        For EXP runs — a nested dict mapping shift string → variable name
        → folder path string::

            {"0.003": {"z": ".../hist/0.003/z", ...}, ...}
    """
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
    """Save a histogram dict as a compressed NPZ file.

    File names follow the convention ``{sym_}{var}_hist_RG{rg_step}.npz``
    where the ``sym_`` prefix is added only when *sym* is ``True`` and
    *var* is ``"z"``.

    Args:
        var: Observable variable name (e.g. ``"z"``, ``"t"``). Used in
            the filename and to decide whether to prepend ``"sym_"``.
        sym: When ``True`` and *var* is ``"z"``, the ``"sym_"`` prefix is
            added to indicate the z-distribution has been symmetrised.
        folder_name: Directory path in which to write the NPZ file.
        rg_step: Current RG iteration index, used in the filename.
        data: Dictionary of arrays passed as keyword arguments to
            :func:`numpy.savez_compressed`.

    Side effects:
        Writes ``{folder_name}/{sym_}{var}_hist_RG{rg_step}.npz`` to disk.
    """
    if sym and var == "z":
        sym_text = "sym_"
    else:
        sym_text = ""
    filename = f"{folder_name}/{sym_text}{var}_hist_RG{rg_step}.npz"

    np.savez_compressed(filename, **data)


def qshe_sampler(
    samples: int,
    rng: np.random.Generator,
    hist_dict: dict,
    y_var: Optional[str] = None,
) -> tuple:
    """Draw joint (z, y_var) samples from a 2-D histogram via conditional resampling.

    Args:
        samples: Number of samples to draw.
        rng: NumPy :class:`~numpy.random.Generator` instance (PCG64) for
            the random draws.
        hist_dict: Dictionary containing at least a ``"2d"`` key mapping
            to a 2-D histogram dict as produced by
            :func:`source.utilities.build_2d_hist`.
        y_var: Name of the second variable used to select the histogram
            key ``z_{y_var}``.  Defaults to ``"mix"`` when ``None``,
            giving key ``"z_mix"``.

    Returns:
        Tuple ``(sample1, sample2)`` where *sample1* is the z-axis draw
        and *sample2* is the y_var-axis draw, each a 1-D float64 array
        of length *samples*.
    """
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

    # --- Build CLI argument parser: base args + QSHE-specific args ---
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

    # --- Load YAML config, apply CLI overrides, build typed dataclass ---
    config = handle_config(args_dict["config"], args.override)
    rg_config = build_config(config)
    rg_config.type = args_dict["type"]

    vars = rg_config.vars

    # --- Resolve output directory and save the finalised config ---
    if args.out is None:
        base_output_dir = build_default_output_dir(config, args.q)
    else:
        base_output_dir = Path(args.out)
    output_dir = base_output_dir / args_dict["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    cur_date = get_current_date()
    print(f"[{cur_date}]: Starting simulation.")

    # --- Redirect stdout and stderr into per-run log files ---
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

    # --- Print config summary (written to the redirected log) ---
    print_config(rg_config, args.q)

    # print(f" Output folders: {json.dumps(output_folders, indent=2)} ")
    # print("-" * 100)

    # --- Resolve the FP distribution file path used to seed EXP runs ---
    starting_t = args.t
    starting_phi = args.phi
    starting_th = args.q
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

    # --- Dispatch to the QSHE workflow (FP or EXP) ---
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

    # --- Restore stdout/stderr and report completion ---
    sys.stdout = orig_output
    sys.stderr = orig_err
    output_file.close()
    error_file.close()
    print(f"Outputs printed to {output_dir}. ")
    end_time = time()
    print(
        f"[{cur_date}]: Simulation completed after {end_time - start_time:.3f} seconds. "
    )
