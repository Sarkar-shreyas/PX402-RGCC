"""Run an RG workflow locally"""

import numpy as np
import sys
from pathlib import Path
from source.parse_config import build_parser, get_project_root
from source.config import (
    handle_config,
    save_updated_config,
    get_nested_data,
    build_config,
)
from source.utilities import (
    convert_t_to_z,
    convert_z_to_t,
    generate_constant_array,
    extract_t_samples,
    generate_initial_t_distribution,
    generate_random_phases,
    rg_data_workflow,
    build_rng,
    center_z_distribution,
    launder,
    get_density,
    save_data,
)


def build_default_output_dir(config: dict, run_type: str) -> Path:
    """Parse the input config dict and build the default output path"""
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root()

    return root / "Local data" / version_str / run_type / "config"


def build_hist(data: np.ndarray, bins: int, range: tuple) -> dict:
    """Constructs a dictionary of histogram values, bin edges, bin centers and densities, then returns it"""
    hist, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    densities = get_density(hist, edges)
    return {"hist": hist, "edges": edges, "centers": centers, "densities": densities}


if __name__ == "__main__":
    # Build parser and parse ocnfig
    parser = build_parser()
    args = parser.parse_args()
    config = handle_config(args.config, args.override)

    if args.out is None:
        output_dir = build_default_output_dir(config, args.type)
    else:
        output_dir = Path(args.out)
    t_data_folder = output_dir / "hist/t"
    z_data_folder = output_dir / "hist/z"
    output_dir.mkdir(parents=True, exist_ok=True)
    t_data_folder.mkdir(parents=True, exist_ok=True)
    z_data_folder.mkdir(parents=True, exist_ok=True)
    save_updated_config(output_dir, config)

    # Change stdout and stderr to other files for logging
    output_file = f"{output_dir}/output.txt"
    error_file = f"{output_dir}/error.txt"
    orig_output = sys.stdout
    orig_err = sys.stderr
    sys.stdout = open(output_file, "w")
    sys.stderr = open(error_file, "w")

    # Load relevant config vars
    rg_config = build_config(config)
    samples = rg_config.samples
    steps = rg_config.steps
    method = rg_config.method
    expr = rg_config.expr
    resample = rg_config.resample
    symmetrise = rg_config.symmetrise
    seed = rg_config.seed
    matrix_batch_size = rg_config.matrix_batch_size
    t_bins = rg_config.t_bins
    t_range = rg_config.t_range
    z_bins = rg_config.z_bins
    z_range = rg_config.z_range
    rng = build_rng(seed)
    if method == "analytic":
        i = 4
    else:
        i = 8
    initial_t = generate_initial_t_distribution(samples, rng)
    ts = extract_t_samples(initial_t, samples, rng)
    phases = generate_random_phases(samples, rng, i)
    # initial_t = generate_constant_array(samples, 1 / np.sqrt(2))
    # phases = generate_constant_array(samples, 0)

    # Main rg loop
    for step in range(steps):
        tprime = rg_data_workflow(method, ts, phases, samples, expr)
        z = convert_t_to_z(tprime)
        t_data = build_hist(tprime, t_bins, t_range)
        z_data = build_hist(z, z_bins, z_range)
        if symmetrise == 1:
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
