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
    var_names = ["r", "t", "tau", "f", "z", "sym_z"]
    var_names = ["r", "t", "tau", "f", "z"]
    z_vars = ["z", "sym_z"]
    other_vars = ["r", "t", "tau", "f"]
    hist_dir = f"{data_folder}/{version}/{rg_config.type}/hist"
    stats_dir = f"{data_folder}/{version}/{rg_config.type}/stats"
    plots_dir = f"{data_folder}/{version}/{rg_config.type}/plots"
    r_folder = f"{hist_dir}/r"
    t_folder = f"{hist_dir}/t"
    tau_folder = f"{hist_dir}/tau"
    f_folder = f"{hist_dir}/f"
    z_folder = f"{hist_dir}/z"
    folder_names = {
        "hist": hist_dir,
        "stats": stats_dir,
        "plots": plots_dir,
        "r": r_folder,
        "t": t_folder,
        "tau": tau_folder,
        "f": f_folder,
        "z": z_folder,
    }
    for folder in folder_names:
        os.makedirs(folder_names[folder], exist_ok=True)
    print("Folders created")

    # Load histogram data
    data_map = defaultdict(list)
    for var in var_names:
        for i in range(num_rg):
            if var == "sym_z":
                file = f"{folder_names['z']}/{var}_hist_RG{i}.npz"
            else:
                file = f"{folder_names[var]}/{var}_hist_RG{i}.npz"
            counts, bins, centers = load_hist_data(file)
            densities = get_density(counts, bins)
            data_map[var].append([counts, bins, centers, densities])
    print("All histogram datasets have been loaded")

    # Plot the other 3 variables without clipping bounds
    for var in var_names:
        if var == "z":
            continue
        filename = f"{plots_dir}/{var}_histogram.png"
        plot_data(var, filename, data_map[var], rg_config.type.upper(), num_rg)

    z_filename = f"{plots_dir}/z_histogram.png"
    fig, (ax0, ax1) = plt.subplots(1, 2, num="z", figsize=(12, 6))
    ax0.set_title("Histogram of z")
    ax0.set_xlabel("z")
    ax0.set_ylabel("Q(z)")
    ax1.set_title("Clipped Histogram of z")
    ax1.set_xlabel("z")
    ax1.set_ylabel("Q(z)")
    ax0.set_xlim((-25.0, 25.0))
    ax1.set_xlim((-5.0, 5.0))
    for i in range(num_rg):
        x_data = data_map["z"][i][2]
        y_data = data_map["z"][i][3]
        ax0.plot(x_data, y_data, label=f"RG{i}")
        ax1.scatter(x_data[::100], y_data[::100], label=f"RG{i}")
    ax0.legend(loc="upper left")
    ax1.legend(loc="upper left")
    fig.savefig(z_filename, dpi=150)
    plt.close(fig)

    print(f"Plots for {var_names} have been made")
    print("-" * 100)
    construct_moments_dict(stats_dir, plots_dir, var_names, data_map, num_rg)
    # print("-" * 100)
    # print(data_map["sym_z"][0][1])
    print("Analysis done.")
