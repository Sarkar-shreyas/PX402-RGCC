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
    """Creates folders for the input version if they don't already exist"""
    version_folder = f"{data_dir}/{version}"
    fp_folder = f"{version_folder}/FP"
    exp_folder = f"{version_folder}/EXP"
    os.makedirs(version_folder, exist_ok=True)
    os.makedirs(fp_folder, exist_ok=True)
    os.makedirs(exp_folder, exist_ok=True)
    return [version_folder, fp_folder, exp_folder]


# ---------- SSH connection and command execution ---------- #
def run_commands(commands: list) -> None:
    print("Running: ", " ".join(commands))
    subprocess.run(commands, check=True)


# ---------- Main driver ---------- #
def transfer_files(args) -> None:
    """Process input args and run a sequence of commands"""

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
                local = f"{taskfarm_dir}/scripts"
            else:
                raise ValueError(f"Invalid push dir entered: {dir}")
            current_commands.extend([local, remote])
            run_commands(current_commands)
        commands = ["scp", "-r"]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    transfer_files(args)
