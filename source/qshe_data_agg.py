"""This file reads existing q-p trial data, and combines them into a single file"""

import math
from pathlib import Path
import numpy as np
import sys
from source.config import get_rg_config
from time import time

if __name__ == "__main__":
    # Load input params
    if len(sys.argv) == 7:
        num_samples = int(sys.argv[1].strip())
        num_steps = int(sys.argv[2].strip())
        q_block_size = int(sys.argv[4].strip())
        output_dir = sys.argv[5].strip()
        vars = [v.strip() for v in sys.argv[6].split(",")]
    else:
        raise SystemExit(
            "Usage: qshe_data_agg.py NUM_SAMPLES NUM_STEPS Q_BLOCK_SIZE OUTPUT_DIR VARS"
        )

    rg_config = get_rg_config()
    if rg_config.model != "qshe":
        raise SystemExit(f"Invalid model {rg_config.model}.")
    if rg_config.metric == "all":
        met_dim = 2
    else:
        met_dim = 1
    print("Beginning aggregation step")
    print("-" * 100)
    start = time()
    block_num = math.ceil(rg_config.q_num / q_block_size)
    output_path = Path(output_dir)
    for var in vars:
        agg_filename = output_path / f"{var}_data_agg.npy"
        final_data = np.empty(
            shape=(rg_config.q_num, rg_config.p_num, num_steps, met_dim),
            dtype=np.float64,
        )
        for block in range(block_num):
            block_path = output_path / f"q{block}"
            if not (block_path / "DONE").exists():
                raise RuntimeError(f"Block {block} incomplete.")

            end_index = min((block + 1) * q_block_size, rg_config.q_num)
            trial_data_file = (
                f"{output_dir}/q{block}/{var}_data_q{block}_{num_samples}_samples.npy"
            )
            trial_data = np.load(trial_data_file)
            final_data[block * q_block_size : end_index, :, :, :] = trial_data
        np.save(agg_filename, final_data)
        print(f"Trial data for {var} aggregated after {time() - start:.3f} seconds")
    print("-" * 100)
    print(
        f"All aggregation steps completed for {block_num} q blocks after {time() - start:.3f} seconds"
    )
