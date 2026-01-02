"""
Loads yaml and sets up configuration
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass
from functools import lru_cache


# ---------- RG Config dataclass ---------- #
@dataclass
class RGConfig:
    version: str
    id: str
    type: str
    output_folder: str
    model: str
    method: str
    resample: str
    expr: str
    symmetrise: int
    seed: int
    steps: int
    samples: int
    matrix_batch_size: int
    inputs: list
    outputs: list
    shifts: list
    z_bins: int
    z_range: tuple
    z_min: float
    z_max: float
    t_bins: int
    t_range: tuple
    t_min: float
    t_max: float
    msd_tol: float
    std_tol: float


def build_config(config: dict) -> RGConfig:
    """Parses a config dict and returns an RGConfig object"""
    version = str(check_required_info(config, "main.version")).strip().lower()
    id = check_required_info(config, "main.id")
    type = check_required_info(config, "main.type")
    output_folder = check_required_info(config, "main.output_folder")
    model = str(check_required_info(config, "engine.model")).strip().lower()
    method = str(check_required_info(config, "engine.method")).strip().lower()
    resample = str(check_required_info(config, "engine.resample")).strip().lower()
    expr = str(check_required_info(config, "engine.expr")).strip().lower()
    symmetrise = int(get_nested_data(config, "engine.symmetrise", 1))
    seed = int(check_required_info(config, "rg_settings.seed"))
    steps = int(check_required_info(config, "rg_settings.steps"))
    samples = int(check_required_info(config, "rg_settings.samples"))
    matrix_batch_size = int(
        check_required_info(config, "rg_settings.matrix_batch_size")
    )
    inputs = check_required_info(config, "data_settings.inputs")
    outputs = check_required_info(config, "data_settings.outputs")
    shift_config = get_nested_data(
        config, "data_settings.shifts", [0.003, 0.005, 0.007, 0.009]
    )
    shifts = [float(str(shift).strip()) for shift in shift_config]
    z_bins = int(get_nested_data(config, "parameter_settings.z.bins", 200))
    z_range = tuple(get_nested_data(config, "parameter_settings.z.range", [0.0, 1.0]))
    z_min = float(z_range[0])
    z_max = float(z_range[1])
    t_bins = int(get_nested_data(config, "parameter_settings.tprime.bins", 200))
    t_range = tuple(
        get_nested_data(config, "parameter_settings.tprime.range", [0.0, 1.0])
    )
    t_min = float(t_range[0])
    t_max = float(t_range[1])
    msd_tol = float(get_nested_data(config, "convergence.msd_tol", 1.0e-3))
    std_tol = float(get_nested_data(config, "convergence.std_tol", 5.0e-4))

    return RGConfig(
        version=version,
        id=id,
        type=type,
        output_folder=output_folder,
        model=model,
        method=method,
        resample=resample,
        expr=expr,
        symmetrise=symmetrise,
        seed=seed,
        steps=steps,
        samples=samples,
        matrix_batch_size=matrix_batch_size,
        inputs=inputs,
        outputs=outputs,
        shifts=shifts,
        z_bins=z_bins,
        z_range=z_range,
        z_min=z_min,
        z_max=z_max,
        t_bins=t_bins,
        t_range=t_range,
        t_min=t_min,
        t_max=t_max,
        msd_tol=msd_tol,
        std_tol=std_tol,
    )


@lru_cache(maxsize=1)
def get_rg_config() -> RGConfig:
    """Load the config file from an env var, parse into an RGConfig object, and store in the lru cache"""
    config_path = os.environ.get("RG_CONFIG")
    if not config_path:
        raise RuntimeError("RG_CONFIG could not be found")
    config = load_yaml(config_path)
    rg_config = build_config(config)
    return rg_config


# ---------- Core yaml interactions ---------- #
def load_yaml(path: str | Path) -> dict:
    """
    Loads the yaml file from the given path into a dictionary
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a dictionary")

    return data


def dump_yaml(data: dict, path: str | Path) -> None:
    """
    Dumps existing data into a yaml file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


# ---------- Access config data from dict ---------- #
def get_nested_data(config: dict, path: str, default: Any = None) -> Any:
    """Returns the nested dictionary from a given path, if present"""
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
    """Checks for required fields in the input config"""
    info = get_nested_data(config, path, None)
    if info is None:
        raise KeyError(f"Missing required config field: {path}")
    return info


# ---------- Handle overrides from the CLI ---------- #
def parse_overrides(input_overrides: list[str]) -> dict:
    """Parses a dictionary of override commands and assigns the correct values to the corresponding setting"""

    overrides = {}
    for pair in input_overrides:
        if "=" not in pair:
            raise ValueError(f"Invalid override command, missing '=': {pair}")
        var, value = pair.split("=", 1)
        var = var.strip()
        value = value.strip()
        if "[" in value:
            parts = value[1:-1].split(",")
            value = parts
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
    """Updates the input config in place with any overrides, and returns the new config. Optionally uses a copy instead if deep = False"""
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
    """Loads the existing config and checks for manual overrides"""
    config = load_yaml(config_file)
    if input_overrides is not None:
        overrides = parse_overrides(input_overrides)
        config = update_config(config, overrides, deep)
    return config


# ---------- File I/O ---------- #
def save_updated_config(run_dir: str | Path, conf: dict) -> None:
    """Save updated config yaml file to the run directory"""
    conf_path = Path(run_dir) / "updated_config.yaml"
    dump_yaml(conf, conf_path)
    # print(conf_path)
