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
import os


# ---------- Helper functions ---------- #
def get_project_root(outer_dirs: int = 2) -> Path:
    """Return the root directory for this project"""
    # from fyp/code/source -> fyp
    return Path(__file__).resolve().parents[outer_dirs]


def get_default_output_dir(config: dict, run_type: str) -> Path:
    """Parse the input config dict and build the default output path"""
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root()

    return root / "job_outputs" / version_str / run_type / "config"


# ---------- Parsing helpers ---------- #
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
        default=None,
        help="Override config settings. Eg; --set 'rg_settings.steps = 5' 'engine.method = numerical'",
    )
    parser.add_argument("--out", default=None, help="Output path for config")

    return parser


def validate_input(input_args) -> dict:
    """Validates input parser args"""
    args_dict = {}
    # Validate config path
    config_path = str(input_args.config).strip()
    if "." not in config_path:
        config_path += ".yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file at {config_path}")
    args_dict["config"] = config_path

    # Check and validate output path
    if input_args.out is not None:
        output_path = str(input_args.out).strip()
        if not os.path.isdir(output_path):
            raise FileNotFoundError(f"Could not find output directory {output_path}")
        args_dict["out"] = output_path
    else:
        args_dict["out"] = None
    # Check and validate type input
    rg_type = str(input_args.type).strip().upper()
    if rg_type not in ("FP", "EXP"):
        raise ValueError(f"Invalid RG type {rg_type} entered, expected 'FP' or 'EXP'.")
    args_dict["type"] = rg_type
    return args_dict


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args_dict = validate_input(args)
    config = handle_config(args_dict["config"], args.override)

    if args.out is None:
        output_dir = get_default_output_dir(config, args_dict["type"])
    else:
        output_dir = Path(args.out)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_updated_config(output_dir, config)
