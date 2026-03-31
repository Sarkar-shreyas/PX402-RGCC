"""CLI entry point for loading, overriding, and saving a validated YAML config.

Two-step flow
-------------
1. **Load & override** — :func:`source.config.handle_config` reads the YAML file and
   applies any ``--set`` key=value pairs on top of it, producing an updated raw dict.
2. **Save** — :func:`source.config.save_updated_config` writes the merged dict back to
   ``updated_config.yaml`` in the resolved output directory, so the exact config used
   for a run is always recorded alongside its outputs.

``--set`` dot-notation syntax
------------------------------
Each ``--set`` argument must be a single string of the form::

    section.subsection.key=value

Keys follow the same dot-separated hierarchy as the YAML file.  Values are parsed by
``yaml.safe_load``, so strings, ints, floats, booleans, and lists are all accepted::

    --set "rg_settings.steps=9"
    --set "engine.method=numerical"
    --set "data_settings.shifts=[0.003, 0.005, 0.007]"

Multiple overrides can be chained in a single invocation::

    --set "rg_settings.steps=9" "engine.method=numerical"

Missing intermediate keys are created automatically; existing leaf values are
overwritten.  The resolution logic lives in :func:`source.config.parse_overrides`
and :func:`source.config.update_config`.

Usage example (from CLAUDE.md)
-------------------------------
::

    python -m source.parse_config \\
        --config Taskfarm/configs/iqhe.yaml \\
        --set "engine.method=numerical" \\
        --out /tmp/configs

This writes ``/tmp/configs/updated_config.yaml`` with the method override applied.
If ``--out`` is omitted the output is placed under
``{project_root}/job_outputs/{version}_{method}_{expr}/{type}/config/``.
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
    """Return the repository root directory by walking up from this file.

    Args:
        outer_dirs: Number of parent directories to ascend from this file's
            location.  The default of ``2`` walks from
            ``{root}/source/parse_config.py`` up to ``{root}``.

    Returns:
        Absolute :class:`~pathlib.Path` to the project root.
    """
    # from fyp/code/source -> fyp
    return Path(__file__).resolve().parents[outer_dirs]


def get_default_output_dir(config: dict, run_type: str) -> Path:
    """Build the default output path for the saved config file.

    Constructs the path as::

        {project_root}/job_outputs/{version}_{method}_{expr}/{run_type}/config

    where ``version``, ``method``, and ``expr`` are read from *config*.

    Args:
        config: Raw config dictionary (after overrides have been applied).
        run_type: Run type string, e.g. ``"FP"`` or ``"EXP"``.  Used as a
            subdirectory level so FP and EXP outputs remain separate.

    Returns:
        Absolute :class:`~pathlib.Path` for the config output directory.
        The directory is not created by this function.
    """
    version = str(get_nested_data(config, "main.version"))
    method = str(get_nested_data(config, "engine.method"))
    expr = str(get_nested_data(config, "engine.expr")).strip().lower()
    # Combine the three identifiers into a single directory name so that
    # different (version, method, expr) combinations never share an output path.
    version_str = f"{version}_{method}_{expr}"

    root = get_project_root()

    return root / "job_outputs" / version_str / run_type / "config"


# ---------- Parsing helpers ---------- #
def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the config validation CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` with ``--config``,
        ``--set``, and ``--out`` arguments registered.  A ``--type`` argument
        is added by the ``__main__`` block before parsing so it is available
        only when the module is run directly.
    """
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
    """Validate parsed CLI arguments and return a normalised args dictionary.

    Checks that the config file exists (appending ``.yaml`` if no extension is
    present), that the output directory exists when ``--out`` is supplied, and
    that ``--type`` is one of the recognised run types.

    Args:
        input_args: Namespace returned by :meth:`argparse.ArgumentParser.parse_args`.
            Expected attributes: ``config`` (str), ``out`` (str or None),
            ``type`` (str).

    Returns:
        Dictionary with keys:

        - ``"config"`` (:class:`str`) — validated path to the YAML config file.
        - ``"out"`` (:class:`str` or ``None``) — validated output directory path,
          or ``None`` if ``--out`` was not supplied.
        - ``"type"`` (:class:`str`) — upper-cased run type (``"FP"``, ``"EXP"``,
          or ``"QP"``).

    Raises:
        FileNotFoundError: If the config file path does not exist, or if
            ``--out`` is supplied but does not point to an existing directory.
        ValueError: If ``--type`` is not one of ``"FP"``, ``"EXP"``, or ``"QP"``.
    """
    args_dict = {}
    # Validate config path
    config_path = str(input_args.config).strip()
    # Allow the user to omit the .yaml extension for convenience
    if "." not in config_path:
        config_path += ".yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file at {config_path}")
    args_dict["config"] = config_path

    # Check and validate output path
    if input_args.out is not None:
        output_path = str(input_args.out).strip()
        # The directory must already exist; we do not create it here so that
        # callers are explicit about where outputs will land.
        if not os.path.isdir(output_path):
            raise FileNotFoundError(f"Could not find output directory {output_path}")
        args_dict["out"] = output_path
    else:
        args_dict["out"] = None
    # Check and validate type input
    rg_type = str(input_args.type).strip().upper()
    if rg_type not in ("FP", "EXP", "QP"):
        raise ValueError(
            f"Invalid RG type {rg_type} entered, expected 'FP, 'EXP' or 'QP'."
        )
    args_dict["type"] = rg_type
    return args_dict


if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument("--type", default="FP", help="The type of RG flow")
    args = parser.parse_args()
    args_dict = validate_input(args)
    config = handle_config(args_dict["config"], args.override)

    if args.out is None:
        output_dir = get_default_output_dir(config, args_dict["type"])
    else:
        output_dir = Path(args.out)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_updated_config(output_dir, config)
