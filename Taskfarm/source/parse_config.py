"""
This file will parse config.yaml files and CLI overrides to return an updated config file
"""

import argparse
from pathlib import Path
from source.config import (
    handle_config,
    get_nested_data,
    save_updated_config,
)


def get_project_root() -> Path:
    """Return the root directory for this project"""
    # from fyp/code/source -> fyp
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the parser for the local RG engine"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="The path to the .yaml config file"
    )
    parser.add_argument(
        "--set",
        dest="override",
        nargs="+",
        action="extend",
        default=[],
        help="Override config settings. Eg; --set 'rg_settings.steps = 5' 'engine.method = numerical'",
    )
    parser.add_argument("--out", default=None, help="Output path for config")
    parser.add_argument(
        "--type", required=True, choices=["FP", "EXP"], help="Type of RG workflow"
    )

    return parser


def get_default_output_dir(config: dict, run_type: str) -> Path:
    """Parse the input config dict and build the default output path"""
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()[0]
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root()

    return root / "job_outputs" / version_str / run_type / "config"


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    config = handle_config(args.config, args.override)

    if args.out is None:
        output_dir = get_default_output_dir(config, args.type)
    else:
        output_dir = Path(args.out)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_updated_config(output_dir, config)
