import matplotlib.pyplot as plt
import os
from collections import defaultdict
from constants import data_dir, config_file, local_dir
from source.config import build_config, load_yaml
from analysis.data_plotting import (
    build_plot_parser,
    build_config_path,
    load_hist_data,
    get_density,
    plot_data,
    construct_moments_dict,
    plot_3d,
)


if __name__ == "__main__":
    parser = build_plot_parser()
    args = parser.parse_args()
    if os.path.exists(args.loc):
        config_path = build_config_path(args.loc, args.version, args.mode)
        if str(args.loc).strip().lower() == "remote":
            data_folder = data_dir
        elif str(args.loc).strip().lower() == "local":
            data_folder = local_dir
    else:
        config_path = str(config_file)
        data_folder = data_dir
    config = load_yaml(config_path)
    print(f"Config loaded from {config_path}")
    rg_config = build_config(config)
    # Load constants

    version = str(args.version)
    num_rg = int(args.steps)
    var_names = ["r", "t", "tau", "f", "loss", "z", "sym_z"]
    # var_names = ["r", "t", "tau", "f", "z"]
    z_vars = ["z", "sym_z"]
    other_vars = ["r", "t", "tau", "f"]
    hist_dir = f"{data_folder}/{version}/{rg_config.type}/hist"
    stats_dir = f"{data_folder}/{version}/{rg_config.type}/stats"
    plots_dir = f"{data_folder}/{version}/{rg_config.type}/plots"
    r_folder = f"{hist_dir}/r"
    t_folder = f"{hist_dir}/t"
    tau_folder = f"{hist_dir}/tau"
    f_folder = f"{hist_dir}/f"
    loss_folder = f"{hist_dir}/loss"
    z_folder = f"{hist_dir}/z"
    folder_names = {
        "hist": hist_dir,
        "stats": stats_dir,
        "plots": plots_dir,
        "r": r_folder,
        "t": t_folder,
        "tau": tau_folder,
        "f": f_folder,
        "loss": loss_folder,
        "z": z_folder,
    }
    for folder in folder_names:
        os.makedirs(folder_names[folder], exist_ok=True)
    print("Folders created")

    # Load histogram data
    data_map = defaultdict(list)
    missing_vars = []
    for var in var_names:
        for i in range(num_rg):
            try:
                if var == "sym_z":
                    file = f"{folder_names['z']}/{var}_hist_RG{i}.npz"
                else:
                    file = f"{folder_names[var]}/{var}_hist_RG{i}.npz"
                counts, bins, centers = load_hist_data(file)
                densities = get_density(counts, bins)
                data_map[var].append([counts, bins, centers, densities])
            except FileNotFoundError:
                missing_vars.append(var)
    print("All histogram datasets have been loaded")

    # Plot the other 3 variables without clipping bounds
    for var in var_names:
        if var in missing_vars:
            continue
        filename = f"{plots_dir}/{var}_histogram.png"
        plot_data(var, filename, data_map[var], rg_config.type.upper(), num_rg)

    if "z" in missing_vars:
        z_vars = ["sym_z"]
    elif "sym_z" in missing_vars:
        z_vars = ["z"]
    else:
        z_vars = ["z", "sym_z"]

    for z_var in z_vars:
        z_filename = f"{plots_dir}/{z_var}_histogram.png"
        fig, (ax0, ax1) = plt.subplots(1, 2, num=f"{z_var}", figsize=(12, 6))
        ax0.set_title(f"Histogram of {z_var}")
        ax0.set_xlabel(f"{z_var}")
        ax0.set_ylabel(f"Q({z_var})")
        ax1.set_title(f"Clipped Histogram of {z_var}")
        ax1.set_xlabel(f"{z_var}")
        ax1.set_ylabel(f"Q({z_var})")
        ax0.set_xlim((-25.0, 25.0))
        ax1.set_xlim((-5.0, 5.0))
        for i in range(num_rg):
            x_data = data_map[z_var][i][2]
            y_data = data_map[z_var][i][3]
            ax0.plot(x_data, y_data, label=f"RG{i}")
            if z_var == "z":
                ax1.scatter(x_data[::100], y_data[::100], label=f"RG{i}")
            else:
                ax1.scatter(x_data, y_data, label=f"RG{i}")
        ax0.legend(loc="upper left")
        ax1.legend(loc="upper left")
        fig.savefig(z_filename, dpi=150)
        plt.close(fig)

    # zf_filename = f"{hist_dir}/zf/zf_hist"
    # zf_plot = f"{plots_dir}/3d_plot.png"
    # plot_3d(zf_filename, num_rg, zf_plot)
    # print(f"3d plot saved to {zf_plot}")

    present_vars = []
    for var in var_names:
        if var in missing_vars:
            continue
        else:
            present_vars.append(var)

    print(f"Plots for {var_names} have been made")
    print("-" * 100)
    construct_moments_dict(stats_dir, plots_dir, present_vars, data_map, num_rg)
    # print("-" * 100)
    # print(data_map["sym_z"][0][1])
    print("Analysis done.")
