from constants import data_dir, config_file, local_dir
from analysis.data_plotting import (
    calculate_average_nu,
    build_plot_parser,
    build_config_path,
)
from source.config import build_config, load_yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import os


def main():
    """
    Main entry point for plotting RG statistics and ν extraction.

    This script detects whether a config file is present (new-format run) or not (legacy run).
    - If config is present, it parses run parameters from the config YAML.
    - If config is absent, it falls back to legacy heuristics (parsing version/steps from folder names or filenames).
    - Loads stats and plots summary figures for the specified RG steps.

    Args:
        None. Uses CLI arguments (see build_plot_parser for options).

    Returns:
        None. Side effects: writes plots to output folder.

    Raises:
        ValueError: If start/end steps are invalid.
        FileNotFoundError: If data files are missing.

    Notes:
        - Output file is written as PNG to the version's output folder.
        - If config is missing, uses folder/filename parsing for legacy runs.
        - Assumption: old-format data uses legacy naming conventions.
    """
    parser = build_plot_parser()
    parser.add_argument("--start", default=0, help="Start step for nu averaging")
    parser.add_argument("--end", default=0, help="End step for nu averaging")
    args = parser.parse_args()
    if os.path.exists(args.loc):
        config_path = build_config_path(args.loc, args.version, args.mode)
    else:
        config_path = str(config_file)
    config = load_yaml(config_path)
    print(f"Config loaded from {config_path}")

    if args.loc == "local":
        data_folder = local_dir
    else:
        data_folder = data_dir
    rg_config = build_config(config)
    num_rg = int(args.steps)
    num_samples = rg_config.samples
    version = str(args.version)
    filename = f"{data_folder}/{version}/overall_stats.json"
    plots_filename = f"{data_folder}/{version}/overall_stats.png"
    print(f"Loading data from {filename}")
    with open(filename, "r", encoding="utf-8") as file:
        stats = json.load(file)
    data = pd.read_json(filename, orient="index")
    start = int(args.start)
    end = int(args.end)
    if start > num_rg:
        raise ValueError(
            f"Start cannot be larger than the number of steps. Got start = {start}, steps = {num_rg}"
        )
    if start > end:
        raise ValueError(
            f"Start step cannot be larger than the end step. Got start = {start}, end = {end}"
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), num="Stats")
    fig.suptitle(
        f"Stats for {version} using {num_samples} samples and RG steps {start + 1}-{end + 1}"
    )
    ax1.set_title("Peak R2")
    ax2.set_title("Mean R2")
    ax3.set_title("Peak Nu")
    ax4.set_title("Mean Nu")
    ax1.set_xlabel("RG Step")
    ax2.set_xlabel("RG Step")
    ax3.set_xlabel("RG Step")
    ax4.set_xlabel("RG Step")

    x = []
    for i in range(start + 1, end + 1):
        x.append(f"RG{i}")
    print(data["Peak Nu"].iloc[start:])
    ax1.scatter(x, data["Peak R2"].iloc[start:end])
    ax2.scatter(x, data["Mean R2"].iloc[start:end])
    ax3.scatter(x, data["Peak Nu"].iloc[start:end])
    ax4.scatter(x, data["Mean Nu"].iloc[start:end])
    calculate_average_nu(stats, start + 1, end + 1)
    plt.savefig(plots_filename, dpi=150)
    plt.close("Stats")
    print(f"Stats plotted and saved to {plots_filename}")


if __name__ == "__main__":
    main()
