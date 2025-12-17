import os
import shutil

from constants import data_dir, CURRENT_VERSION


def transfer_code():
    """Function that transfers code files to the remote directory"""
    pass


def transfer_scripts():
    """Function that transfers script files to the remote directory"""
    pass


def retrieve_fp_data():
    """Function that retrieves data for the FP analysis from the remote directory"""
    pass


def retrieve_exp_data():
    """Function that retrieves data from the EXP analysis from the remote directory"""
    pass


if __name__ == "__main__":
    os.makedirs(f"{data_dir}/v{CURRENT_VERSION}/FP", exist_ok=True)
    os.makedirs(f"{data_dir}/v{CURRENT_VERSION}/EXP", exist_ok=True)
