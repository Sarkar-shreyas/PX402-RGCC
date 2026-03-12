"""
This file performs the q-p trials for an input array of q and p values
"""

import numpy as np
from pathlib import Path
import shutil
import sys
import os
from source.config import get_rg_config
from source.utilities import build_rng, qp_trials, generate_random_phases
from time import time

if __name__ == "__main__":
    # Load input params
    if len(sys.argv) == 8:
        num_samples = int(sys.argv[1].strip())
        num_steps = int(sys.argv[2].strip())
        q_block = int(sys.argv[3].strip())
        q_block_size = int(sys.argv[4].strip())
        phi_seed = int(sys.argv[5].strip())
        gen_seed = int(sys.argv[6].strip())
        output_dir = sys.argv[7].strip()
    else:
        raise SystemExit(
            "Usage: qshe_data_gen.py NUM_SAMPLES NUM_STEPS Q_BLOCK Q_BLOCK_SIZE PHI_SEED GEN_SEED OUTPUT_DIR"
        )

    # Load config and set up default params
    phi_rng = build_rng(phi_seed)
    gen_rng = build_rng(gen_seed)
    rg_config = get_rg_config()
    if rg_config.model != "qshe":
        raise SystemExit(f"Invalid model {rg_config.model}.")

    batch_size = rg_config.matrix_batch_size
    if batch_size > num_samples:
        batch_size = num_samples
    # p_vals = np.linspace(rg_config.p_min, rg_config.p_max, rg_config.p_num)
    if rg_config.metric == "all":
        met_dim = 3
    else:
        met_dim = 1

    # Setup temp and shared output folders
    temp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    local_block_dir = temp_dir / f"q{q_block}"
    shared_block_dir = Path(output_dir) / f"q{q_block}"

    # In case there's an old existing dir from failed runs
    if local_block_dir.exists():
        shutil.rmtree(local_block_dir)
    local_block_dir.mkdir(parents=True, exist_ok=False)

    if shared_block_dir.exists():
        raise RuntimeError(f"{shared_block_dir} folder already exists")

    print("-" * 100)
    print(f"Beginning q-p trials for q = {q_block}")
    start = time()
    q_start = rg_config.q_min + q_block * q_block_size
    q_end = min((q_block + 1) * q_block_size, rg_config.q_num)
    qsep = 0.5 / (rg_config.q_num - 1)
    qs = np.linspace(q_start * qsep, q_end * qsep, q_block_size)
    ps = np.linspace(rg_config.p_min, rg_config.p_max, rg_config.p_num)
    p_trial_data = np.empty(
        shape=(len(qs), rg_config.p_num, num_steps, met_dim), dtype=np.float64
    )
    q_trial_data = np.empty(
        shape=(len(qs), rg_config.p_num, num_steps, met_dim), dtype=np.float64
    )

    phis = generate_random_phases(num_samples, phi_rng, 16)
    assert len(qs) <= q_block_size
    for i, q in np.ndenumerate(qs):
        for j, p in np.ndenumerate(ps):
            a, b = qp_trials(
                q,
                p,
                num_samples,
                num_steps,
                phis,
                gen_rng,
                rg_config.metric,
                rg_config.fixed,
                rg_config.outputs,
                rg_config.inputs,
                batch_size,
            )
            p_trial_data[i[0], j[0], :, :] = a
            q_trial_data[i[0], j[0], :, :] = b
            if j[0] % 100 == 0:
                print(f"Trial no. {j[0]} completed in {time() - start:.3f} seconds.")

    p_filename = f"p_data_q{q_block}_{num_samples}_samples.npy"
    q_filename = f"q_data_q{q_block}_{num_samples}_samples.npy"
    np.save(local_block_dir / p_filename, p_trial_data)
    np.save(local_block_dir / q_filename, q_trial_data)

    # Add a flag to check if the job was successful
    (local_block_dir / "DONE").touch()

    # Move data from temp folder back to shared FS
    shutil.move(str(local_block_dir), str(shared_block_dir))

    print(
        f"q-p trial for q = {q_block} completed and saved to {output_dir} after {time() - start:.3f} seconds"
    )
    print("-" * 100)
