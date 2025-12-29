"""
This file will parse config.yaml files and CLI overrides to return an updated config file
"""

import argparse
from source.config import (
    handle_config,
    save_updated_config,
)


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
        help="Override config settings. Eg; --set rg.total_steps = 5 rng.seed = 231",
    )
    parser.add_argument("--out", required=True, help="Output path for config")
    parser.add_argument("--task", type=int, default=0, help="Task ID for array jobs")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    config = handle_config(args.config, args.override)
    save_updated_config(args.out, config)
