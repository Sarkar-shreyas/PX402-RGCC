"""Local IQHE FP/EXP driver for development and testing.

Purpose
-------
Single-process driver for running IQHE Renormalization Group (RG) workflows
locally, intended for development and testing only — not for production use.
Re-uses the same ``source/`` library code as the Taskfarm HPC scripts but runs
at reduced sample counts (typically 32 M samples, 7 steps) in one Python
process without Slurm job arrays.

Differences from HPC Taskfarm scripts
--------------------------------------
- **Sample count**: 32 M samples locally vs 320 M–480 M on the cluster.
- **RG steps**: 7 locally vs 9 on the cluster.
- **No Slurm**: all computation is sequential in one process; HPC uses
  parallelised job arrays coordinated by ``Taskfarm/scripts/run_rg.sh`` and
  ``Taskfarm/scripts/run_shifts.sh``.
- **No aggregation step**: the Taskfarm pipeline splits data generation and
  histogram construction across many tasks; this driver combines both in a
  single loop.

CLI Usage
---------
Run from the repository root::

    python -m Local.run_local_iqhe \\
        --config Local/configs/local_iqhe \\
        --set "rg_settings.steps=3" "rg_settings.samples=10000000" \\
        --type FP

    python -m Local.run_local_iqhe \\
        --config Local/configs/local_iqhe \\
        --set "rg_settings.steps=3" "rg_settings.samples=10000000" \\
        --type EXP

Output Location
---------------
Outputs are written under ``Local data/`` in the repository root::

    Local data/{version}_{method}_{expr}/FP/
        hist/t/t_hist_RG{i}.npz
        hist/z/z_sym_hist_RG{i}.npz   (or z_hist_RG{i}.npz when symmetrise=0)
        output.txt
        error.txt
        output_locs.json
        updated_config.yaml

    Local data/{version}_{method}_{expr}/EXP/
        {shift}/hist/t/t_hist_RG{i}.npz
        {shift}/hist/z/z_hist_RG{i}.npz

Notes
-----
- stdout and stderr are redirected to ``output.txt`` and ``error.txt`` inside
  the chosen output directory when executed as a script.
- The EXP workflow seeds its initial distribution from the last FP-step
  z-histogram; run the FP workflow before EXP for a given config.
"""

from time import time
import numpy as np
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
    generate_initial_t_distribution,
    generate_random_phases,
    get_current_date,
    rg_data_workflow,
    build_rng,
    center_z_distribution,
    launder,
    get_density,
    save_data,
)
from constants import T_DICT, PHI_DICT

# ---------- Helper utility ---------- #


def build_default_output_dir(config: dict) -> Path:
    """Build the default local output directory for a config.

    Args:
        config: Parsed configuration dictionary (result of
            :func:`source.parse_config.validate_input` /
            :func:`source.config.handle_config`). Must include ``main.version``
            and ``engine.method`` keys. ``engine.expr`` is also used to form
            the directory name.

    Returns:
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

    Args:
        data: 1-D array of samples to histogram.
        bins: Number of histogram bins.
        range: ``(min, max)`` binning range.

    Returns:
        Dictionary with keys: ``hist`` (counts), ``edges`` (bin edges),
        ``centers`` (bin centers) and ``densities`` (density per bin computed
        using :func:`source.utilities.get_density`).
    """
    hist, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    densities = get_density(hist, edges)
    return {"hist": hist, "edges": edges, "centers": centers, "densities": densities}


def _apply_symmetrise(
    symmetrise: int,
    z_data: dict,
    t_data: dict,
    samples: int,
    rng,
    resample,
) -> tuple[str, np.ndarray]:
    """Apply symmetrise / no-symmetrise branch and return (sym_prefix, t_sample).

    Args:
        symmetrise: Flag from config — 1 to symmetrise, 0 to skip.
        z_data: Dict with keys 'hist', 'edges', 'centers' for the z-histogram.
        t_data: Dict with keys 'hist', 'edges', 'centers' for the t-histogram.
        samples: Number of samples to draw when laundering.
        rng: NumPy Generator instance.
        resample: Resampling mode string passed to :func:`source.utilities.launder`.

    Returns:
        ``(sym, t_sample)`` where ``sym`` is ``"sym_"`` or ``""`` and
        ``t_sample`` is the laundered 1-D array of t amplitudes.
    """
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
    return sym, t_sample


def print_config(config: RGConfig) -> None:
    """Print a compact, human-readable summary of the main run settings.

    Args:
        config: Configuration dataclass returned by
            :func:`source.config.build_config`.

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
    if config.type.strip().upper() == "EXP":
        shifts = config.shifts
        shifts_str = ", ".join(str(s) for s in shifts)
        p("Shifts", shifts_str)
    print("-" * len(header))


# ---------- Main RG drivers ---------- #


def rg_fp(
    rg_config: RGConfig, output_folders: dict, starting_t: int, starting_phi: int
) -> dict:
    """Run an FP (fixed-point) RG workflow locally and write histograms.

    Performs ``rg_config.steps`` RG iterations. Each step:

    1. Applies the RG map via :func:`source.utilities.rg_data_workflow` to
       produce the next-step amplitude samples ``t'``.
    2. Converts ``t'`` to the log-ratio ``z`` via
       :func:`source.utilities.convert_t_to_z`.
    3. Builds t- and z-histograms.
    4. Optionally symmetrises the z-distribution (when
       ``rg_config.symmetrise == 1``) to enforce particle-hole symmetry.
    5. Launders the histogram back into a new sample array for the next step.
    6. Writes NPZ histogram files to the ``t`` and ``z`` sub-folders of
       ``output_folders``.

    Args:
        rg_config: Configuration dataclass containing numeric settings
            (samples, bins, ranges, resampling behaviour, seed, etc.).
        output_folders: Mapping with keys ``'t'`` and ``'z'`` giving output
            directories for t- and z-histograms respectively.
        starting_t: If non-zero, selects a fixed starting t value from
            ``T_DICT`` (keyed by the ``--t`` CLI argument). Zero means draw
            from a flat distribution.
        starting_phi: If non-zero, selects a fixed starting phase from
            ``PHI_DICT`` (keyed by the ``--phi`` CLI argument). Zero means
            draw random phases.

    Returns:
        Mapping of step identifiers to the generated NPZ file paths, e.g.
        ``{"RG0": {"t": "...", "z": "..."}, ...}``.

    Notes:
        The implementation references an external ``args`` variable when
        constructing constant initial arrays if ``starting_t`` or
        ``starting_phi`` is non-zero. This variable is provided when the
        module is executed as a script; if you call :func:`rg_fp`
        programmatically you must supply ``starting_t``/``starting_phi``
        values accordingly (see the module-level ``if __name__ == '__main__'``
        block).
    """
    # --- Unpack run settings from the validated config dataclass ---
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
    rng = build_rng(seed)
    t_data_folder = output_folders["t"]
    z_data_folder = output_folders["z"]
    # Analytic method uses 4 loop phases; numerical (matrix) uses 8.
    if method == "analytic":
        i = 4
    else:
        i = 8

    # --- Build initial sample arrays (t amplitudes and phases) ---
    if starting_t != 0:
        initial_t = generate_constant_array(samples, T_DICT[f"{args.t}"])
    else:
        initial_t = generate_initial_t_distribution(samples, rng)

    if starting_phi != 0:
        phases = generate_constant_array(samples, PHI_DICT[f"{args.phi}"], i)
    else:
        phases = generate_random_phases(samples, rng, i)
    ts = extract_t_samples(initial_t, samples, rng)
    # initial_t = generate_constant_array(samples, 1 / np.sqrt(2))
    # phases = generate_constant_array(samples, 0)
    output_files = {}

    # --- Main RG iteration loop ---
    for step in range(steps):
        print(f" Proceeding with RG step {step}. ")

        # Apply the RG map to produce next-step amplitudes t', then convert to z
        tprime = rg_data_workflow(method, ts, phases, samples, expr, batch_size)
        z = convert_t_to_z(tprime)

        # Build histograms for both t and z representations
        t_data = build_hist(tprime, t_bins, t_range)
        z_data = build_hist(z, z_bins, z_range)

        # Symmetrisation branch: fold z-distribution about zero to enforce
        # particle-hole symmetry, then launder back to t samples.
        # Unsymmetrised branch: launder directly from the t-histogram.
        sym, t_sample = _apply_symmetrise(
            symmetrise, z_data, t_data, samples, rng, resample
        )

        # Resample from the laundered distribution for the next RG step
        ts = extract_t_samples(t_sample, samples, rng)

        # Write histogram NPZ files for this step
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

    Loads the last FP-run z-histogram, launders an initial sample from it,
    then for each shift value in ``rg_config.shifts``:

    1. Translates every resampled z-value by the shift: ``z' = z + shift``.
    2. Maps the shifted z to amplitude ``t'`` via
       :func:`source.utilities.convert_z_to_t`.
    3. Runs ``rg_config.steps`` RG iterations (same per-step logic as
       :func:`rg_fp`).
    4. Writes NPZ histogram files per shift and per step.

    The growth of the mean z across shifts and steps is used by the analysis
    scripts (``analysis/critical_exponent.py``) to extract the critical
    exponent ν.

    Args:
        rg_config: Configuration dataclass with samples, bins, ranges, shifts
            and other resampling parameters.
        output_folders: Mapping that, for each shift value (as a string key),
            provides a sub-dict with ``'t'`` and ``'z'`` keys giving the
            corresponding output directory paths.
        fp_dist: Path to a fixed-point NPZ file (containing keys
            ``'histval'``, ``'binedges'`` and ``'bincenters'``). Loaded to
            construct the laundered initial distribution that each shift is
            applied to.
        starting_phi: If non-zero, a constant phase array is used; otherwise
            phases are generated randomly from the RNG.

    Returns:
        Nested mapping of shift → step → file paths, e.g.
        ``{"0.003": {"RG0": {"t": "...", "z": "..."}}, ...}``.

    Side effects:
        Writes NPZ files to disk via :func:`source.utilities.save_data` and
        prints progress to stdout.
    """
    # --- Unpack run settings from the validated config dataclass ---
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
    # Analytic method uses 4 loop phases; numerical (matrix) uses 8.
    if method == "analytic":
        i = 4
    else:
        i = 8
    output_files = {}

    # --- Load the FP fixed-point z-histogram and launder an initial sample ---
    fp_data = np.load(fp_dist)
    fp_hist = fp_data["histval"]
    fp_edges = fp_data["binedges"]
    fp_centers = fp_data["bincenters"]
    initial_z = launder(samples, fp_hist, fp_edges, fp_centers, rng, resample)

    # --- Iterate over each shift value ---
    for shift in shifts:
        t_data_folder = output_folders[f"{shift}"]["t"]
        z_data_folder = output_folders[f"{shift}"]["z"]

        # Apply constant shift to z, then convert to t amplitude for RG input
        shifted_z = initial_z + shift
        shifted_t = convert_z_to_t(shifted_z)
        if starting_phi != 0:
            phases = generate_constant_array(samples, starting_phi, i)
        else:
            phases = generate_random_phases(samples, rng, i)
        ts = extract_t_samples(shifted_t, samples, rng)

        # --- RG iteration loop for this shift ---
        for step in range(steps):
            print(f" Proceeding with RG step {step} of shift {shift}. ")

            # Apply the RG map to produce next-step amplitudes t', then convert to z
            tprime = rg_data_workflow(method, ts, phases, samples, expr, batch_size)
            z = convert_t_to_z(tprime)

            # Build histograms for both t and z representations
            t_data = build_hist(tprime, t_bins, t_range)
            z_data = build_hist(z, z_bins, z_range)

            # Symmetrisation branch: fold z-distribution about zero, then launder
            # back to t samples. Unsymmetrised branch: launder from t-histogram.
            sym, t_sample = _apply_symmetrise(
                symmetrise, z_data, t_data, samples, rng, resample
            )
            ts = extract_t_samples(t_sample, samples, rng)

            # Write histogram NPZ files for this shift and step
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

    # --- Build CLI argument parser and parse arguments ---
    parser = build_parser()
    parser.add_argument(
        "--type",
        required=True,
        default="FP",
        choices=["FP", "EXP"],
        help="Type of RG workflow",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=0,
        help="Use a constant value for the t array. 1: t=0, 2: t=1/2, 3: t=1/sqrt(2), 4: t=1",
    )
    parser.add_argument(
        "--phi",
        type=int,
        default=0,
        help="Enter a constant value for the phi array. 1: phi=0, 2: phi=pi/4, 3: phi=pi/2, 4: phi=pi, 5: phi=2pi",
    )
    args = parser.parse_args()
    args_dict = validate_input(args)

    # --- Load YAML config and apply any CLI overrides, then build typed dataclass ---
    config = handle_config(args_dict["config"], args.override)
    rg_config = build_config(config)

    # --- Resolve output directory and save the finalised config for reproducibility ---
    if args.out is None:
        base_output_dir = build_default_output_dir(config)
    else:
        base_output_dir = Path(args.out)
    output_dir = base_output_dir / args_dict["type"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    # --- Redirect stdout and stderr into per-run log files ---
    output_filename = f"{output_dir}/output.txt"
    error_filename = f"{output_dir}/error.txt"
    orig_output = sys.stdout
    orig_err = sys.stderr
    output_file = open(output_filename, "w")
    error_file = open(error_filename, "w")
    sys.stdout = output_file
    sys.stderr = error_file

    # --- Create per-variable output sub-folders for histogram files ---
    print_config(rg_config)
    output_folders = {}
    if args_dict["type"] == "EXP":
        # EXP runs write one folder tree per shift value
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
        # FP runs write a single pair of t/z folders
        t_data_folder = output_dir / "hist/t"
        z_data_folder = output_dir / "hist/z"
        t_data_folder.mkdir(parents=True, exist_ok=True)
        z_data_folder.mkdir(parents=True, exist_ok=True)
        output_folders.update({"t": str(t_data_folder), "z": str(z_data_folder)})

    print(f" Output folders: {json.dumps(output_folders, indent=2)} ")
    print("-" * 100)

    # --- Dispatch to FP or EXP workflow ---
    starting_t = args.t
    starting_phi = args.phi
    # EXP seeds its initial distribution from the last FP z-histogram
    fp_data_file = f"{base_output_dir}/FP/hist/z/z_sym_hist_RG{rg_config.steps - 1}.npz"
    if args_dict["type"] == "FP":
        hist_outputs = rg_fp(rg_config, output_folders, starting_t, starting_phi)
    else:
        hist_outputs = rg_exp(rg_config, output_folders, fp_data_file, starting_phi)
    print("-" * 100)

    # --- Restore stdout/stderr and write the output-file location manifest ---
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
