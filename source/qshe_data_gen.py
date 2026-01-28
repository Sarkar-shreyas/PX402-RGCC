import os
import sys
import numpy as np
from Local.run_local_qshe import build_hist
from QSHE.testing_qshe import gen_initial_data, numerical_solver
from source.utilities import (
    generate_random_phases,
    build_rng,
    convert_t_to_geff,
    convert_g_to_z,
    build_2d_hist,
)
from source.config import get_rg_config


if __name__ == "__main__":
    # Load input params, checking if we're starting RG steps or continuing from an input sample
    if len(sys.argv) == 7:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        seed = int(sys.argv[5].strip())
        f_val = float(sys.argv[6].strip())
        existing_t_file = "None"
    elif len(sys.argv) == 8:
        array_size = int(sys.argv[1].strip())
        output_dir = sys.argv[2].strip()
        initial = int(sys.argv[3].strip())
        rg_step = int(sys.argv[4].strip())
        seed = int(sys.argv[5].strip())
        existing_f_file = sys.argv[6].strip()
        existing_t_file = sys.argv[7].strip()
    else:
        raise SystemExit(
            "Usage: data_generation.py ARRAY_SIZE OUTPUT_DIR INITIAL RG_STEP SEED F_VAL|F_FILE [EXISTING_T_FILE]"
        )

    print("-" * 100)
    print(f"Beginning data generation for RG step {rg_step}")
    rng = build_rng(seed)
    rg_config = get_rg_config()
    inputs = rg_config.inputs
    batch_size = rg_config.matrix_batch_size
    t_bins = rg_config.t_bins
    t_range = rg_config.t_range
    z_bins = rg_config.z_bins
    z_range = rg_config.z_range
    z_2d_bins = z_bins // 10
    t_2d_bins = t_bins // 10
    if initial == 1:
        starting_t = 0
        starting_phi = 0
        initial_data = gen_initial_data(
            array_size, starting_t, starting_phi, f_val, rng
        )
        t = initial_data["t"]
        f = initial_data["f"]
        phases = initial_data["phi"]
        print("Generated initial distributions")
    else:
        print(f"Using t and f data from {existing_t_file}")
        t = np.load(existing_t_file)
        loss = np.load(existing_f_file)
        f = np.sqrt(loss)
        phases = generate_random_phases(array_size, rng, 16)
    indexes = rng.integers(0, array_size, size=(array_size, 5))
    ts = np.take(t, indexes)
    fs = np.take(f, indexes)
    tprime = numerical_solver(ts, fs, phases, array_size, 2, inputs, batch_size)
    rprime = numerical_solver(ts, fs, phases, array_size, 9, inputs, batch_size)
    tauprime = numerical_solver(ts, fs, phases, array_size, 10, inputs, batch_size)
    fprime = numerical_solver(ts, fs, phases, array_size, 17, inputs, batch_size)
    g_eff = convert_t_to_geff(tprime, rprime)
    z = convert_g_to_z(g_eff)
    loss = (
        np.abs(np.reshape(tauprime, shape=array_size)) ** 2
        + np.abs(np.reshape(fprime, shape=array_size)) ** 2
    )
    t_data = build_hist(tprime, t_bins, t_range)
    r_data = build_hist(rprime, t_bins, t_range)
    tau_data = build_hist(tauprime, t_bins, t_range)
    f_data = build_hist(fprime, t_bins, t_range)
    loss_data = build_hist(loss, t_bins, t_range)
    z_data = build_hist(z, z_bins, z_range)
    os.makedirs(output_dir, exist_ok=True)
    if rg_config.symmetrise == 1:
        hist_dict = build_2d_hist(z, loss, z_2d_bins, t_2d_bins, z_range, t_range, True)
    else:
        hist_dict = build_2d_hist(z, loss, z_2d_bins, t_2d_bins, z_range, t_range)

    output_file = os.path.join(output_dir, f"data_hist_RG{rg_step}.npz")
    np.savez_compressed(
        output_file,
        zfcounts=hist_dict["zf"]["counts"],
        zfdensities=hist_dict["zf"]["densities"],
        rcounts=r_data["counts"],
        tcounts=t_data["counts"],
        taucounts=tau_data["counts"],
        fcounts=f_data["counts"],
        losscounts=loss_data["counts"],
        zcounts=z_data["counts"],
        redges=r_data["edges"],
        tedges=t_data["edges"],
        tauedges=tau_data["edges"],
        fedges=f_data["edges"],
        lossedges=loss_data["edges"],
        zedges=z_data["edges"],
        rcenters=r_data["centers"],
        tcenters=t_data["centers"],
        taucenters=tau_data["centers"],
        fcenters=f_data["centers"],
        losscenters=loss_data["centers"],
        zcenters=z_data["centers"],
        rdensities=r_data["densities"],
        tdensities=t_data["densities"],
        taudensities=tau_data["densities"],
        fdensities=f_data["densities"],
        lossdensities=loss_data["densities"],
        zdensities=z_data["densities"],
    )

    print(f"Parameter hists generated for RG step {rg_step} and saved to {output_file}")
    # if existing_t_file is not None and os.path.exists(existing_t_file):
    #     # Delete old files once done to prevent buildup
    #     os.remove(existing_t_file)
    print("-" * 100)
