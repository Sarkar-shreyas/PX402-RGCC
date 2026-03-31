"""Configuration loading, validation, and typed dataclasses for the RG pipeline.

Purpose
-------
This module is the single source of truth for run configuration.  It converts a raw
nested dictionary (typically read from a YAML file) into a validated, typed dataclass
that all downstream modules consume.  Callers should never parse YAML or read
environment variables for config themselves — import from here instead.

Exported dataclasses
--------------------
- :class:`BaseConfig` — fields shared by every run type
- :class:`IQHEConfig` — extends BaseConfig with IQHE-specific histogram and shift
  parameters
- :class:`QSHEConfig` — extends BaseConfig with QSHE-specific q-p sweep parameters
- ``RGConfig`` — ``Union[IQHEConfig, QSHEConfig]`` type alias

Primary entry point
-------------------
:func:`build_config` — accepts a raw config dict, dispatches on ``engine.model``, and
returns the appropriate typed dataclass.

Secondary utilities
-------------------
- :func:`handle_config` — load a YAML file and apply optional CLI overrides in one call
- :func:`load_yaml` / :func:`dump_yaml` — low-level YAML I/O
- :func:`get_rg_config` — cached singleton config for the current process (reads
  ``RG_CONFIG`` from the environment)
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Literal, Union
from dataclasses import dataclass
from functools import lru_cache


# ---------- Additional helpers ---------- #
def _check_lowercase_keys(data: dict, parent: str = "") -> None:
    """
    Recursively check that all dictionary keys are lowercase.

    Parameters
    ----------
    data : dict
        Dictionary to check.
    parent : str, optional
        Parent key path for error reporting (default: '').

    Raises
    ------
    ValueError
        If a key is not lowercase.
    """
    for key, val in data.items():
        if isinstance(key, str) and key != key.lower():
            raise KeyError(f"Key {parent}{key} must be all lowercase")
        if isinstance(val, dict):
            _check_lowercase_keys(val, parent=f"{parent}{key}.")


# ---------- RG Config dataclasses ---------- #


@dataclass
class BaseConfig:
    """Fields shared by every RG run, regardless of model type.

    Attributes:
        version: Run identifier string (e.g. ``"fp_iqhe"``).  Combined with
            ``method`` and ``expr`` to form the output-directory path
            ``{version}/{method}/{expr}``.
        id: Short job or array ID used for HPC bookkeeping; varies per submission.
        type: Run mode — ``"fp"`` for a fixed-point run or ``"exp"`` for a
            perturbation (shifted) run.
        output_folder: Root output path for this run, constructed as
            ``{version}/{method}/{expr}``.
        model: Physical model being simulated; one of ``"iqhe"`` or ``"qshe"``.
        method: RG transformation method — ``"analytic"`` (4-phase) or
            ``"numerical"`` (8-phase, matrix).
        expr: Mathematical expression variant for the RG map (e.g. ``"shaw"``).
        seed: NumPy PCG64 RNG seed for reproducibility.  Typical values: 1234
            (local), 12345 (HPC).
        steps: Number of RG iterations per run.  Typical: 7 (local), 9 (HPC).
        samples: Total MC samples per run.  Typical: 32 000 000 (local),
            480 000 000 (HPC).
        matrix_batch_size: Number of samples processed per matrix-multiplication
            batch; keeps memory usage bounded.  Typical: 100 000.
        inputs: Initial parameter values fed into the first RG step.
            Typical: ``[1.0, 0.0]``.
        msd_tol: Mean-squared-displacement convergence tolerance; the run is
            considered converged when the MSD between successive z-distributions
            falls below this value.  Default: 1e-3.
        std_tol: Standard-deviation convergence tolerance; complementary to
            ``msd_tol``.  Default: 5e-4.
    """

    # Main params
    version: str
    id: str
    type: str
    output_folder: str

    # Engine params
    model: Literal["iqhe", "qshe"]
    method: str
    expr: str

    # rg flow settings
    seed: int
    steps: int
    samples: int
    matrix_batch_size: int

    # Basic data settings
    inputs: list[float]
    msd_tol: float
    std_tol: float


@dataclass
class IQHEConfig(BaseConfig):
    """Configuration for an Integer Quantum Hall Effect (IQHE) run.

    Extends :class:`BaseConfig` with histogram resolution, z-distribution
    symmetrisation, EXP perturbation shifts, and output-variable selection.

    Attributes:
        model: Overrides the base field; always ``"iqhe"`` for this dataclass.
        resample: Resampling strategy for the inverse-CDF step.  ``"i"``
            selects inverse-CDF resampling.
        symmetrise: Whether to symmetrise the z-distribution after each RG
            step.  ``0`` = off, ``1`` = on.  Symmetrisation enforces
            particle-hole symmetry around z = 0.
        shifts: List of small perturbation magnitudes applied to the FP
            distribution in EXP runs, used to measure the growth rate of the
            relevant RG eigenvalue and hence extract ν.
            Typical: ``[0.003, 0.005, 0.007, 0.009]``.
        outputs: Integer codes selecting which observables to record.
            Typical: ``[8]``.
        z_bins: Number of histogram bins for the z log-ratio distribution.
            Typical: 50 000.
        z_range: ``(min, max)`` range of the z histogram.
            Typical: ``(-25.0, 25.0)``.
        z_min: Lower bound of ``z_range``; unpacked for convenience.
        z_max: Upper bound of ``z_range``; unpacked for convenience.
        t_bins: Number of histogram bins for the t amplitude distribution.
            Typical: 1 000.
        t_range: ``(min, max)`` range of the t histogram.  Always
            ``(0.0, 1.0)`` since t is a probability amplitude.
        t_min: Lower bound of ``t_range``; unpacked for convenience.
        t_max: Upper bound of ``t_range``; unpacked for convenience.
    """

    # IQHE-specific engine and rg flow settings
    model: Literal["iqhe"]
    resample: str
    symmetrise: int
    shifts: list[float]
    outputs: list[int]

    # IQHE-specific var settings
    z_bins: int
    z_range: tuple
    z_min: float
    z_max: float
    t_bins: int
    t_range: tuple
    t_min: float
    t_max: float


@dataclass
class QSHEConfig(BaseConfig):
    """Configuration for a Quantum Spin Hall Effect (QSHE) run.

    Extends :class:`BaseConfig` with a 2-D (q, p) parameter sweep and
    per-trial observable selection.

    Attributes:
        model: Overrides the base field; always ``"qshe"`` for this dataclass.
        metric: Aggregation statistic computed over each q-p grid cell.
            One of ``"mean"``, ``"median"``, or ``"all"`` (returns all
            per-trial values).
        fixed: Whether to fix one sweep parameter during the run.
            ``0`` = both q and p vary, ``1`` = one is held fixed.
        vars: Observable names to record per trial.  Used in local FP runs
            only.  Typical: ``["r", "t", "tau", "f", "g", "surv", "z",
            "mix", "p"]``.
        outputs: Output variable names written to the result arrays.
        q_range: ``(q_min, q_max)`` sweep range for the q parameter.
            Typical: ``(0.0, 1.0)``.
        q_min: Lower bound of ``q_range``; unpacked for convenience.
        q_max: Upper bound of ``q_range``; unpacked for convenience.
        q_num: Number of grid points along the q axis.  Typical: 50.
        p_range: ``(p_min, p_max)`` sweep range for the p parameter.
            Typical: ``(0.0, 1.0)``.
        p_min: Lower bound of ``p_range``; unpacked for convenience.
        p_max: Upper bound of ``p_range``; unpacked for convenience.
        p_num: Number of grid points along the p axis.  Typical: 50.
    """

    # QSHE-specific engine params
    model: Literal["qshe"]
    metric: Literal["mean", "median", "std", "all"]
    fixed: Literal[0, 1]

    # QSHE-specific parameter settings
    vars: list[str]
    outputs: list[str]
    q_range: tuple
    q_min: float
    q_max: float
    q_num: int
    p_range: tuple
    p_min: float
    p_max: float
    p_num: int


RGConfig = Union[IQHEConfig, QSHEConfig]


def build_config(config: dict) -> RGConfig:
    """Parse a config dictionary and return a validated RGConfig dataclass.

    Dispatches on ``config["engine"]["model"]`` to construct either an
    :class:`IQHEConfig` or a :class:`QSHEConfig`.  All required fields are
    extracted via :func:`check_required_info`; optional fields fall back to
    sensible defaults via :func:`get_nested_data`.

    Args:
        config: Nested configuration dictionary, typically produced by
            :func:`load_yaml` and optionally mutated by :func:`update_config`.

    Returns:
        An :class:`IQHEConfig` when ``engine.model == "iqhe"``, or a
        :class:`QSHEConfig` when ``engine.model == "qshe"``.

    Raises:
        KeyError: Any field accessed via :func:`check_required_info` is absent
            from *config*.
        ValueError: Raised in three situations —
            (1) ``engine.model`` is neither ``"iqhe"`` nor ``"qshe"``;
            (2) ``rg_settings.metric`` (QSHE) is not one of ``"mean"``,
            ``"median"``, or ``"all"``; or
            (3) ``rg_settings.fixed`` (QSHE) is not ``0`` or ``1``.
        TypeError: A type coercion (``int()``, ``float()``) fails because a
            field value in *config* is not convertible to the expected type.
    """
    # --- Extract base fields shared by all model types ---
    version = str(check_required_info(config, "main.version")).strip().lower()
    id = check_required_info(config, "main.id")
    type = check_required_info(config, "main.type")
    output_folder = check_required_info(config, "main.output_folder")
    model = str(check_required_info(config, "engine.model")).strip().lower()
    method = str(check_required_info(config, "engine.method")).strip().lower()
    expr = str(check_required_info(config, "engine.expr")).strip().lower()
    seed = int(check_required_info(config, "rg_settings.seed"))
    steps = int(check_required_info(config, "rg_settings.steps"))
    samples = int(check_required_info(config, "rg_settings.samples"))
    matrix_batch_size = int(
        check_required_info(config, "rg_settings.matrix_batch_size")
    )
    inputs = check_required_info(config, "data_settings.inputs")
    msd_tol = float(get_nested_data(config, "convergence.msd_tol", 1.0e-3))
    std_tol = float(get_nested_data(config, "convergence.std_tol", 5.0e-4))

    # --- Dispatch on model type: extract model-specific fields and return typed dataclass ---
    if model == "iqhe":
        # --- IQHE-specific fields: histogram resolution, symmetrisation, shifts ---
        shift_config = get_nested_data(
            config, "data_settings.shifts", [0.003, 0.005, 0.007, 0.009]
        )
        z_range = tuple(
            get_nested_data(config, "parameter_settings.z.range", [0.0, 1.0])
        )
        t_range = tuple(
            get_nested_data(config, "parameter_settings.tprime.range", [0.0, 1.0])
        )
        return IQHEConfig(
            version=version,
            id=id,
            type=type,
            output_folder=output_folder,
            model=model,
            method=method,
            expr=expr,
            seed=seed,
            steps=steps,
            samples=samples,
            matrix_batch_size=matrix_batch_size,
            inputs=inputs,
            msd_tol=msd_tol,
            std_tol=std_tol,
            resample=str(check_required_info(config, "engine.resample"))
            .strip()
            .lower(),
            symmetrise=int(get_nested_data(config, "engine.symmetrise", 1)),
            shifts=[float(str(shift).strip()) for shift in shift_config],
            outputs=check_required_info(config, "data_settings.outputs"),
            z_bins=int(get_nested_data(config, "parameter_settings.z.bins", 200)),
            z_range=z_range,
            z_min=float(z_range[0]),
            z_max=float(z_range[1]),
            t_bins=int(get_nested_data(config, "parameter_settings.tprime.bins", 200)),
            t_range=t_range,
            t_min=float(t_range[0]),
            t_max=float(t_range[1]),
        )
    elif model == "qshe":
        # --- QSHE-specific fields: q-p sweep grid, metric, and observable selection ---
        metric = str(check_required_info(config, "rg_settings.metric"))
        # "std" is declared in the Literal but not yet supported by qshe_data_agg
        if metric not in ("mean", "median", "all"):
            raise ValueError(
                f"Metric : {metric} invalid. Must be 'mean', 'median' or 'all'"
            )
        fixed = int(check_required_info(config, "rg_settings.fixed"))
        # YAML may parse 'true'/'false' as booleans; explicit int check prevents silent misuse
        if fixed not in (0, 1):
            raise ValueError(f"fixed : {fixed} invalid. Must be 0 or 1.")
        outputs = check_required_info(config, "data_settings.outputs")
        vars = check_required_info(config, "parameter_settings.vars")
        q_min = float(get_nested_data(config, "parameter_settings.q.min", 0.0))
        q_max = float(get_nested_data(config, "parameter_settings.q.max", 0.5))
        p_min = float(get_nested_data(config, "parameter_settings.p.min", 0.001))
        p_max = float(get_nested_data(config, "parameter_settings.p.max", 0.999))
        q_num = int(get_nested_data(config, "parameter_settings.q.num", 1000))
        p_num = int(get_nested_data(config, "parameter_settings.p.num", 999))
        q_range = (q_min, q_max)
        p_range = (p_min, p_max)

        return QSHEConfig(
            version=version,
            id=id,
            type=type,
            output_folder=output_folder,
            model=model,
            method=method,
            expr=expr,
            seed=seed,
            steps=steps,
            samples=samples,
            matrix_batch_size=matrix_batch_size,
            inputs=inputs,
            msd_tol=msd_tol,
            std_tol=std_tol,
            metric=metric,
            fixed=fixed,
            vars=vars,
            outputs=outputs,
            q_range=q_range,
            q_min=q_min,
            q_max=q_max,
            q_num=q_num,
            p_range=p_range,
            p_min=p_min,
            p_max=p_max,
            p_num=p_num,
        )
    else:
        raise ValueError(f"Model {model} invalid. Must be 'iqhe' or 'qshe'.")


@lru_cache(maxsize=1)
def get_rg_config() -> RGConfig:
    """Return the cached RGConfig singleton for the current process.

    Reads the YAML config path from the ``RG_CONFIG`` environment variable,
    loads and validates it, and caches the result so subsequent calls return
    the same object without re-parsing.

    Returns:
        The validated :class:`RGConfig` instance for this process.

    Raises:
        RuntimeError: If the ``RG_CONFIG`` environment variable is not set.
    """
    config_path = os.environ.get("RG_CONFIG")
    if not config_path:
        raise RuntimeError("RG_CONFIG could not be found")
    config = load_yaml(config_path)
    rg_config = build_config(config)
    return rg_config


# ---------- Core yaml interactions ---------- #
def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file and return its contents as a dictionary.

    Also validates that all keys are lowercase (via :func:`_check_lowercase_keys`)
    to catch accidental capitalisation before it propagates.

    Args:
        path: Path to the YAML file (str or :class:`pathlib.Path`).

    Returns:
        Parsed YAML data as a nested dictionary.  An empty file returns
        an empty dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML file does not parse to a dictionary.
        KeyError: If any dictionary key is not all-lowercase.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a dictionary")
    _check_lowercase_keys(data)
    return data


def dump_yaml(data: dict, path: str | Path) -> None:
    """Serialise a dictionary to a YAML file, creating parent directories as needed.

    Args:
        data: Dictionary to serialise.  Must contain only YAML-safe types.
        path: Output file path (str or :class:`pathlib.Path`).  Parent
            directories are created if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


# ---------- Access config data from dict ---------- #
def get_nested_data(config: dict, path: str, default: Any = None) -> Any:
    """Retrieve a value from a nested config dict using a dot-separated key path.

    Args:
        config: Nested configuration dictionary.
        path: Dot-separated key path, e.g. ``"rg_settings.steps"``.
            The path is lowercased and stripped before use.
        default: Value to return when any key along the path is absent or
            when an intermediate value is not a dict.  Default: ``None``.

    Returns:
        The value at the specified path, or ``default`` if any key is not
        found.
    """
    path = path.strip().lower()
    keys = path.split(".")
    data = config
    for key in keys:
        # print(f"Key {key} found.")
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    # print(f"Data is currently: {data}")
    return data


def check_required_info(config: dict, path: str) -> Any:
    """Return a required field from the config, raising if absent.

    Thin wrapper around :func:`get_nested_data` that converts a ``None``
    return value into a :exc:`KeyError`.

    Args:
        config: Nested configuration dictionary.
        path: Dot-separated key path to the required field.

    Returns:
        The value at the specified path (never ``None``).

    Raises:
        KeyError: If the field is absent or resolves to ``None``.
    """
    info = get_nested_data(config, path, None)
    if info is None:
        raise KeyError(f"Missing required config field: {path}")
    return info


# ---------- Handle overrides from the CLI ---------- #
def parse_overrides(input_overrides: list[str]) -> dict:
    """Parse a list of ``"key=value"`` override strings into a nested dict.

    Each string must contain exactly one ``=`` separator.  The left side is
    split on ``.`` to form nested dict keys; the right side is parsed with
    ``yaml.safe_load`` so that booleans, integers, floats, and lists are
    converted to their native Python types.

    Args:
        input_overrides: List of override strings in the form
            ``"section.key=value"``, e.g. ``["rg_settings.steps=5"]``.

    Returns:
        Nested dictionary that can be merged into a config dict via
        :func:`update_config`.

    Raises:
        ValueError: If any override string lacks an ``=`` character, or if
            the key portion (left of ``=``) is empty.
    """

    overrides = {}
    for pair in input_overrides:
        if "=" not in pair:
            raise ValueError(f"Invalid override command, missing '=': {pair}")
        var, value = pair.split("=", 1)
        var = var.strip()
        value = value.strip()
        value = yaml.safe_load(value)
        if not var:
            raise ValueError(f"Invalid override command, key is empty: {pair}")
        keys = var.split(".")

        temp_overrides = overrides
        for key in keys[:-1]:
            if key not in temp_overrides or not isinstance(temp_overrides[key], dict):
                temp_overrides[key] = {}
            temp_overrides = temp_overrides[key]
        temp_overrides[keys[-1]] = value
    # print(overrides)
    return overrides


def update_config(config: dict, overrides: dict, deep: bool = True) -> dict:
    """Recursively merge ``overrides`` into ``config``.

    For each key in ``overrides``: if both the existing value and the
    override value are dicts, the merge recurses; otherwise the override
    replaces the existing value.

    Args:
        config: Original configuration dictionary (mutated in place when
            ``deep=True``).
        overrides: Dictionary of values to apply.  Nested dicts trigger
            recursive merging.
        deep: When ``True`` (default), operates on ``config`` directly.
            When ``False``, operates on a shallow copy.  Note: the name
            is inverted relative to typical convention — ``True`` means
            in-place, not copy.

    Returns:
        The updated configuration dictionary (same object as ``config``
        when ``deep=True``).
    """
    if deep:
        current_config = config
    else:
        current_config = config.copy()
    # Update the config dict recursively
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(current_config.get(key), dict):
            update_config(current_config[key], val)
        else:
            current_config[key] = val
    return current_config


def handle_config(
    config_file: str | Path,
    input_overrides: Optional[list[str]] = None,
    deep: bool = True,
) -> dict:
    """Load a YAML config file and optionally apply CLI overrides in one call.

    Convenience wrapper that combines :func:`load_yaml`, :func:`parse_overrides`,
    and :func:`update_config`.

    Args:
        config_file: Path to the YAML config file.
        input_overrides: Optional list of ``"key=value"`` override strings
            (as produced by ``--set`` CLI arguments).  Skipped when
            ``None``.  Default: ``None``.
        deep: Forwarded to :func:`update_config`.  Default: ``True``.

    Returns:
        The final (post-override) configuration dictionary.
    """
    config = load_yaml(config_file)
    if input_overrides is not None:
        overrides = parse_overrides(input_overrides)
        config = update_config(config, overrides, deep)
    return config


# ---------- File I/O ---------- #
def save_updated_config(run_dir: str | Path, conf: dict) -> None:
    """Write the finalised config dict to ``updated_config.yaml`` in the run directory.

    Also prints the full path of the written file for logging purposes.

    Args:
        run_dir: Directory under which ``updated_config.yaml`` is written.
            Created by :func:`dump_yaml` if it does not exist.
        conf: Configuration dictionary to serialise.
    """
    conf_path = Path(run_dir) / "updated_config.yaml"
    dump_yaml(conf, conf_path)
    print(conf_path)
