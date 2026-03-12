"""This file reads existing q-p trial data, and combines them into a single file"""

import math
from pathlib import Path
import numpy as np
import sys
from source.config import QSHEConfig, get_rg_config
import json
from time import time


def store_state(
    config: QSHEConfig, output_dir: str | Path, vars: list, block_size: int
):
    n = config.samples
    k = config.steps
    qstep = (config.q_max - config.q_min) / config.q_num
    pstep = (config.p_max - config.p_min) / config.p_num
    fix = config.fixed
    met = config.metric
    state = {
        "q": {
            "Min": config.q_min,
            "Max": config.q_max,
            "Num": config.q_num,
            "Step": qstep,
        },
        "p": {
            "Min": config.p_min,
            "Max": config.p_max,
            "Num": config.p_num,
            "Step": pstep,
        },
        "vars": vars,
        "config": {
            "Samples": n,
            "Steps": k,
            "Fixed": bool(fix),
            "Metric": met,
            "Block size": block_size,
        },
    }
    state_filename = f"{output_dir}/trial_state.json"
    with open(state_filename, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State config saved to {state_filename}")


if __name__ == "__main__":
    # Load input params
    if len(sys.argv) == 6:
        num_samples = int(sys.argv[1].strip())
        num_steps = int(sys.argv[2].strip())
        q_block_size = int(sys.argv[3].strip())
        output_dir = sys.argv[4].strip()
        vars = [v.strip() for v in sys.argv[5].split(",")]
    else:
        raise SystemExit(
            "Usage: qshe_data_agg.py NUM_SAMPLES NUM_STEPS Q_BLOCK_SIZE OUTPUT_DIR VARS"
        )

    rg_config = get_rg_config()
    if rg_config.model != "qshe":
        raise SystemExit(f"Invalid model {rg_config.model}.")
    if rg_config.metric == "all":
        met_dim = 3
    else:
        met_dim = 1
    print("Beginning aggregation step")
    print("-" * 100)
    start = time()
    block_num = math.ceil(rg_config.q_num / q_block_size)
    print(f"Num blocks = {block_num}")
    final_block_size = (block_num * q_block_size) - rg_config.q_num
    output_path = Path(output_dir)
    for var in vars:
        agg_filename = output_path / f"{var}_data_agg.npy"
        final_data = np.empty(
            shape=(rg_config.q_num, rg_config.p_num, num_steps, met_dim),
            dtype=np.float64,
        )
        print(f"Final data shape = {final_data.shape}")
        for block in range(block_num):
            block_path = output_path / f"q{block}"
            if not (block_path / "DONE").exists():
                raise RuntimeError(f"Block {block} incomplete.")
            trial_data_file = (
                f"{output_dir}/q{block}/{var}_data_q{block}_{num_samples}_samples.npy"
            )
            trial_data = np.load(trial_data_file)
            if block == block_num - 1:
                q_block_size = final_block_size
                print(f"Final block {block} has size = {q_block_size}")
                start_index = rg_config.q_num - final_block_size
                end_index = rg_config.q_num
                final_data[start_index:end_index, :, :, :] = trial_data
            else:
                start_index = block * q_block_size
                end_index = min((block + 1) * q_block_size, rg_config.q_num)
                final_data[start_index:end_index, :, :, :] = trial_data
            if block % 5 == 0:
                print(f"Agg done for block {block}.")
        np.save(agg_filename, final_data)
        print(f"Trial data for {var} aggregated after {time() - start:.3f} seconds")
        print(f"Agg data saved to {agg_filename}")
    print("-" * 100)
    store_state(rg_config, output_dir, vars, q_block_size)
    print("-" * 100)
    print(
        f"All aggregation steps completed for {block_num} q blocks after {time() - start:.3f} seconds"
    )
