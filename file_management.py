"""Utilities to transfer code, scripts and job outputs between local repo and remote cluster.

This module implements the canonical staging workflow used by the project. It is
intentionally thin: it constructs and runs either `scp` (Windows) or `rsync`
(Linux/Mac) commands to push local folders (code, scripts, configs) to the
remote host and to pull job outputs back into local storage. The exact remote
paths and the repo->remote mapping are implemented here and relied upon by the
documentation (README/docs).

Important behaviour (implemented in this module):
- Local `source/` is pushed to `<REMOTE_ROOT>/code/` (so remote runtime lives at
    `<REMOTE_ROOT>/code/source/`).
- Local `Taskfarm/scripts/*.sh` and `Taskfarm/configs/*` are pushed to
    `<REMOTE_ROOT>/scripts/`.
- Pulling artifacts retrieves data from `<REMOTE_ROOT>/job_outputs/{version}/{FP|EXP}/...`.

The module reads configuration values from `constants.py` (env-based). Do not
change runtime behaviour in this file without updating the docs which treat it
as the source of truth.
"""

import os
import subprocess
import argparse
from constants import (
    data_dir,
    CURRENT_VERSION,
    host,
    remote_dir,
    taskfarm_dir,
    root_dir,
)


# ---------- Utilities ---------- #
def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the transfer CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the CLI options used by the project's transfer
        helper. Options include `--push`, `--pull`, `--version`, `--type`, and
        `--sys` (controls whether `scp` or `rsync` is used).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=CURRENT_VERSION, help="The version to pull"
    )
    parser.add_argument(
        "--action", default="push", help="Which action to do: Pull or Push"
    )
    parser.add_argument(
        "--push",
        default=[],
        action="append",
        help="Which folder to push: source, scripts, etc.",
    )
    parser.add_argument(
        "--pull",
        default=[],
        action="append",
        help="Which folder to pull: 'hist', 'config', etc.",
    )
    parser.add_argument("--type", default="FP", help="The run mode")
    parser.add_argument(
        "--sys", default="windows", help="The operating system being used"
    )
    parser.add_argument("--step", default=None, help="Which RG step to pull from")
    return parser


def create_local_folders(version: str) -> list:
    """Ensure local destination folders exist for a given version.

    Parameters
    ----------
    version : str
        Version identifier (e.g. `fp_iqhe_numerical_shaw`). The function will
        create `<DATA_DIR>/<version>/`, `<DATA_DIR>/<version>/FP/` and
        `<DATA_DIR>/<version>/EXP/` where `DATA_DIR` is taken from
        `constants.data_dir`.

    Returns
    -------
    list
        A list with three paths: `[version_folder, fp_folder, exp_folder]`.
    """
    version_folder = f"{data_dir}/{version}"
    fp_folder = f"{version_folder}/FP"
    exp_folder = f"{version_folder}/EXP"
    os.makedirs(version_folder, exist_ok=True)
    os.makedirs(fp_folder, exist_ok=True)
    os.makedirs(exp_folder, exist_ok=True)
    return [version_folder, fp_folder, exp_folder]


# ---------- SSH connection and command execution ---------- #
def run_commands(commands: list) -> None:
    """Execute a shell command list and raise on non-zero exit.

    Parameters
    ----------
    commands : list
        The command and arguments to execute (same shape as passed to
        `subprocess.run`). This function prints the command before executing
        it. It will raise `subprocess.CalledProcessError` if the command
        returns a non-zero exit code.
    """
    print("Running: ", " ".join(commands))
    subprocess.run(commands, check=True)


# ---------- Main driver ---------- #
def transfer_files(args) -> None:
    """Perform push or pull actions according to CLI args.

    The function supports two high-level actions controlled by `--action`:

    - `push`: upload local artifacts to the remote host. Supported push
        targets: `code`, `scripts`, `config`.
    - `pull`: retrieve remote job outputs. Supported pull targets include
        `hist` and `config` (the latter pulls `job_outputs/{version}/{type}/config`).

    Parameters
    ----------
    args : argparse.Namespace
            Parsed arguments returned by :func:`build_parser`. Expected attributes
            include `version`, `action`, `push`, `pull`, `type`, `sys`, and
            optionally `step`.

    Raises
    ------
    ValueError
            If an unknown `--sys`, `--type`, `--action`, or push/pull target is
            provided.
    """

    if args.version is not None:
        version = str(args.version).strip().lower()
    else:
        version = CURRENT_VERSION

    folders = create_local_folders(version)
    commands = []
    if str(args.sys).strip().lower() == "windows":
        commands = ["scp", "-r"]
    elif str(args.sys).strip().lower() in ("linux", "mac"):
        commands = ["rsync", "-avz", "--partial", "--progress"]
    else:
        raise ValueError(f"Invalid os name entered: {args.sys}")

    folder_type = str(args.type).strip().upper()
    if folder_type == "FP":
        local = folders[1]
    elif folder_type == "EXP":
        local = folders[2]
    else:
        raise ValueError(f"Invalid RG type entered: {folder_type}")

    if args.step is None:
        rgs = "RG*"
    elif str(args.step).isdigit:
        rgs = f"RG{args.step}"
    else:
        raise ValueError(f"Invalid RG step entered: {args.step}")

    action = str(args.action).strip().lower()
    if action == "pull":
        dirs = args.pull
    elif action == "push":
        dirs = args.push
    else:
        raise ValueError(f"Invalid action {action} entered. Expected 'pull' or 'push'")
    print(f"Running commands for {dirs}")
    for dir in dirs:
        current_commands = commands
        dir = str(dir).strip().lower()
        if action == "pull":
            if dir == "config":
                remote = (
                    f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}/{dir}"
                )
            else:
                remote = f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}/data/{rgs}/{dir}"
            current_commands.extend([remote, local])
            run_commands(current_commands)
        else:
            if dir == "code":
                remote = f"{host}:{remote_dir}/{dir}"
                local = f"{root_dir}/source"
            elif dir == "config":
                remote = f"{host}:{remote_dir}/scripts"
                local = f"{taskfarm_dir}/configs"
            elif dir == "scripts":
                remote = f"{host}:{remote_dir}/{dir}"
                local = f"{taskfarm_dir}/scripts/*.sh"
            else:
                raise ValueError(f"Invalid push dir entered: {dir}")
            current_commands.extend([local, remote])
            run_commands(current_commands)
        commands = ["scp", "-r"]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    transfer_files(args)
