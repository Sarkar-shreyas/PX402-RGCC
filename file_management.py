"""Utilities to transfer code, scripts and job outputs between local repo and remote cluster.

Purpose
-------
This module implements the canonical staging workflow for the RG Monte Carlo pipeline.
It constructs and executes either ``scp`` (Windows) or ``rsync`` (Linux/Mac) commands
to synchronise files between the local workstation and the HPC cluster (vulcan2).

Actions
-------
push
    Uploads local artefacts to the remote host.  Three targets are supported:

    - ``code``    — local ``source/`` → ``<REMOTE_DIR>/code/``
                    (the remote runtime therefore lives at ``<REMOTE_DIR>/code/source/``)
    - ``scripts`` — local ``Taskfarm/scripts/*.sh`` → ``<REMOTE_DIR>/scripts/``
    - ``config``  — local ``Taskfarm/configs/`` → ``<REMOTE_DIR>/scripts/``

pull
    Retrieves job outputs from the remote host.  Two targets are supported:

    - ``hist``   — pulls histogram directories from
                   ``<REMOTE_DIR>/job_outputs/<version>/<type>/data/<RG*>/<dir>``
                   (FP runs) or the ``shift_<shift>`` subtree (EXP runs).
    - ``config`` — pulls the updated config from
                   ``<REMOTE_DIR>/job_outputs/<version>/<type>/config``.

Authentication
--------------
All remote operations use SSH under the hood (via ``scp`` or ``rsync``).
**SSH key-based authentication to vulcan2 must be configured** — the commands
are executed non-interactively and will hang or fail if a password prompt appears.
Ensure your public key is present in ``~/.ssh/authorized_keys`` on the cluster and
that ``~/.ssh/config`` resolves ``vulcan2`` to the correct hostname.

Environment variables (read from ``.env`` via ``constants.py``)
--------------------------------------------------------------
HOST
    Alias for the HPC cluster, e.g. ``vulcan2``.  Used as the ``user@host`` prefix
    in every remote path.
REMOTE_DIR
    Absolute path to the project root on the cluster,
    e.g. ``/storage/physics/phuhjf/fyp``.
DATA_DIR
    Local directory where pulled artefacts are written,
    e.g. ``...\\Data from taskfarm``.

The exact remote paths and the repo→remote mapping are implemented here and relied upon
by the documentation (README/docs). Do not change runtime behaviour in this file without
updating the docs which treat it as the source of truth.
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
from source.config import build_config, load_yaml


# ---------- Utilities ---------- #
def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the transfer CLI.

    Returns:
        argparse.ArgumentParser: Parser configured with the CLI options used by
            the project's transfer helper.  Options:

            --version (str):
                Version identifier that labels the remote output directory,
                e.g. ``fp_iqhe_numerical_shaw``.  Defaults to
                ``constants.CURRENT_VERSION``.
            --action (str):
                Top-level action: ``"push"`` (local → remote) or
                ``"pull"`` (remote → local).  Default: ``"push"``.
            --push (list[str]):
                Repeatable flag; each value names a push target
                (``"code"``, ``"scripts"``, ``"config"``).
            --pull (list[str]):
                Repeatable flag; each value names a pull target
                (``"hist"``, ``"config"``).
            --type (str):
                Run mode that selects the remote subdirectory:
                ``"FP"``, ``"EXP"``, or ``"QP"``.  Default: ``"FP"``.
            --sys (str):
                Operating system of the *local* machine; controls whether
                ``scp`` (``"windows"``) or ``rsync`` (``"linux"``/``"mac"``)
                is used.  Default: ``"windows"``.
            --step (str | None):
                Specific RG step to pull (e.g. ``"3"``).  When ``None``,
                the glob ``RG*`` is used so all steps are retrieved.
            --shift (str | None):
                Shift label used when pulling from an EXP run, e.g. ``"0.003"``.
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
    parser.add_argument(
        "--shift", default=None, help="Which shift to pull, if pulling from EXP"
    )
    return parser


def create_local_folders(version: str, mode: str = "iqhe") -> list:
    """Ensure local destination folders exist for a given version.

    Creates the top-level version directory and the run-mode-specific
    subdirectories under ``constants.data_dir`` so that subsequent ``scp``/
    ``rsync`` pulls have a valid target.

    Args:
        version (str): Version identifier, e.g. ``"fp_iqhe_numerical_shaw"``.
            A directory ``<DATA_DIR>/<version>/`` is created (or left intact
            if it already exists).
        mode (str): Pipeline mode.

            - ``"iqhe"`` (default) — creates ``FP/`` and ``EXP/`` subdirs.
            - ``"qshe"`` — creates a ``QP/`` subdir instead.

    Returns:
        list[str]: Ordered list of created (or verified) paths:

            - IQHE: ``[version_folder, fp_folder, exp_folder]``
            - QSHE: ``[version_folder, qp_folder]``
    """
    # Build the top-level versioned output directory under DATA_DIR.
    version_folder = f"{data_dir}/{version}"
    os.makedirs(version_folder, exist_ok=True)
    folders = [version_folder]
    if mode == "iqhe":
        fp_folder = f"{version_folder}/FP"
        exp_folder = f"{version_folder}/EXP"
        folders.append(fp_folder)
        folders.append(exp_folder)
        os.makedirs(fp_folder, exist_ok=True)
        os.makedirs(exp_folder, exist_ok=True)
    elif mode == "qshe":
        qp_folder = f"{version_folder}/QP"
        os.makedirs(qp_folder, exist_ok=True)
        folders.append(qp_folder)
    return folders


# ---------- SSH connection and command execution ---------- #
def run_commands(commands: list) -> None:
    """Execute a shell command list and raise on non-zero exit.

    Prints the assembled command to stdout before execution so the operator
    can see exactly what ``scp``/``rsync`` invocation is being run, then
    delegates to :func:`subprocess.run` with ``check=True``.

    Args:
        commands (list[str]): The command and its arguments as a flat list,
            in the same format accepted by :func:`subprocess.run`.
            Example: ``["scp", "-r", "user@host:/remote/path", "/local/path"]``.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            return code (i.e. ``scp``/``rsync`` reported an error).
    """
    print("Running: ", " ".join(commands))
    # Execute the transfer command; check=True propagates any non-zero exit as
    # CalledProcessError so failures are visible immediately rather than silently
    # producing an incomplete local copy.
    subprocess.run(commands, check=True)


# ---------- Main driver ---------- #
def transfer_files(args) -> None:
    """Perform push or pull actions according to CLI args.

    Dispatches to either a push (local → remote) or pull (remote → local)
    transfer for each target listed in ``args.push`` / ``args.pull``.

    Path mappings — push
    --------------------
    +------------+---------------------------------------------+------------------------------------------+
    | target     | local source                                | remote destination                       |
    +============+=============================================+==========================================+
    | ``code``   | ``<ROOT_DIR>/source``                       | ``<REMOTE_DIR>/code``                    |
    +------------+---------------------------------------------+------------------------------------------+
    | ``scripts``| ``<TASKFARM_DIR>/scripts/*.sh``             | ``<REMOTE_DIR>/scripts``                 |
    +------------+---------------------------------------------+------------------------------------------+
    | ``config`` | ``<TASKFARM_DIR>/configs``                  | ``<REMOTE_DIR>/scripts``                 |
    +------------+---------------------------------------------+------------------------------------------+

    Path mappings — pull
    --------------------
    +------------+-----------+---------------------------------------------------------------+
    | target     | type      | remote source                                                 |
    +============+===========+===============================================================+
    | ``config`` | any       | ``<REMOTE_DIR>/job_outputs/<version>/<type>/config``          |
    +------------+-----------+---------------------------------------------------------------+
    | ``hist``   | ``QP``    | ``<REMOTE_DIR>/job_outputs/<version>/QP``                     |
    +------------+-----------+---------------------------------------------------------------+
    | ``hist``   | ``FP``    | ``<REMOTE_DIR>/job_outputs/<version>/FP/data/<RG*>/hist``     |
    +------------+-----------+---------------------------------------------------------------+
    | ``hist``   | ``EXP``   | ``<REMOTE_DIR>/job_outputs/<version>/EXP/shift_<shift>/data/<RG*>/hist`` |
    +------------+-----------+---------------------------------------------------------------+

    On Windows ``scp -r`` is used; on Linux/Mac ``rsync -avz --partial --progress``
    is used instead (better resumption and progress reporting for large transfers).

    If the remote path does not exist, ``scp`` will exit with a non-zero code and
    :func:`run_commands` will raise :class:`subprocess.CalledProcessError`.

    Args:
        args (argparse.Namespace): Parsed arguments returned by :func:`build_parser`.
            Expected attributes: ``version``, ``action``, ``push``, ``pull``,
            ``type``, ``sys``, ``step``, ``shift``.

    Raises:
        ValueError: If an unknown value is supplied for ``--sys``, ``--type``,
            ``--action``, or an individual push/pull target.
        subprocess.CalledProcessError: Propagated from :func:`run_commands` if
            the underlying ``scp``/``rsync`` call fails.
    """

    if args.version is not None:
        version = str(args.version).strip().lower()
    else:
        version = CURRENT_VERSION
    folder_type = str(args.type).strip().upper()

    # Create local directories so the pull destination always exists.
    if folder_type == "QP":
        folders = create_local_folders(version, "qshe")
    else:
        folders = create_local_folders(version)

    # Select the transfer command set based on the local OS.
    # scp is used on Windows (no native rsync); rsync is preferred elsewhere
    # because it supports partial transfers and incremental synchronisation.
    commands = []
    if str(args.sys).strip().lower() == "windows":
        commands = ["scp", "-r"]
    elif str(args.sys).strip().lower() in ("linux", "mac"):
        commands = ["rsync", "-avz", "--partial", "--progress"]
    else:
        raise ValueError(f"Invalid os name entered: {args.sys}")

    # Resolve the local destination directory from the folder list returned by
    # create_local_folders.  Index 1 = FP/QP subdirectory; index 2 = EXP subdir.
    if folder_type == "FP":
        local = folders[1]
    elif folder_type == "EXP":
        # Append the shift label so each perturbation lands in its own directory.
        local = f"{folders[2]}/shift{args.shift}"
        # local = folders[2]
    elif folder_type == "QP":
        local = folders[1]
    else:
        raise ValueError(f"Invalid RG type entered: {folder_type}")

    # Build the RG-step glob.  None means "all steps"; a digit selects one step.
    if args.step is None:
        rgs = "RG*"
    elif str(args.step).isdigit():
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
        current_commands = list(commands)
        dir = str(dir).strip().lower()
        if action == "pull":
            if dir == "config":
                # Config lives directly under the run-type directory, not inside data/.
                remote = (
                    f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}/{dir}"
                )
            else:
                if folder_type == "QP":
                    # QP outputs are stored flat under the QP directory.
                    remote = f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}"
                elif folder_type == "FP":
                    # FP: navigate into data/<RG*>/<dir> (e.g. data/RG*/hist).
                    remote = f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}/data/{rgs}/{dir}"
                    # local = f"{data_dir}"
                else:
                    # EXP: outputs are partitioned by shift value under shift_<shift>/data/.
                    remote = f"{host}:{remote_dir}/job_outputs/{version}/{folder_type}/shift_{args.shift}/data/{rgs}/{dir}"
            # Append remote source then local destination to complete the command.
            current_commands.extend([remote, local])
            run_commands(current_commands)
        else:
            if dir == "code":
                # Push the source/ package to the cluster's code/ directory.
                remote = f"{host}:{remote_dir}/{dir}"
                local = f"{root_dir}/source"
            elif dir == "config":
                # Config files go into the remote scripts/ directory alongside the
                # shell scripts so the Slurm jobs can find them by relative path.
                remote = f"{host}:{remote_dir}/scripts"
                local = f"{taskfarm_dir}/configs"
            elif dir == "scripts":
                # Push Slurm shell scripts; the *.sh glob is expanded by the shell.
                remote = f"{host}:{remote_dir}/{dir}"
                local = f"{taskfarm_dir}/scripts/*.sh"
            else:
                raise ValueError(f"Invalid push dir entered: {dir}")
            # Append local source then remote destination to complete the command.
            current_commands.extend([local, remote])
            run_commands(current_commands)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    transfer_files(args)
