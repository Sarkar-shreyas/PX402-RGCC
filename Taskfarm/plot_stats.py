from data_plotting import calculate_average_nu
from constants import NUM_RG, CURRENT_VERSION, DATA_DIR, N
import json
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    filename = f"{DATA_DIR}/v{CURRENT_VERSION}/overall_stats.json"
    plots_filename = f"{DATA_DIR}/v{CURRENT_VERSION}/overall_stats.png"
    print(f"Loading data from {filename}")
    with open(filename, "r", encoding="utf-8") as file:
        stats = json.load(file)
    data = pd.read_json(filename, orient="index")
    start = 1
    # print(type(stats))
    # print(data.head())
    # print(data["Peak R2"])
    # print(data["Peak R2"].iloc[3:])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), num="Stats")
    fig.suptitle(
        f"Stats for v{CURRENT_VERSION} using {N} samples and RG steps {start + 1}-{NUM_RG + 1}"
    )
    ax1.set_title("Peak R2")
    ax2.set_title("Mean R2")
    ax3.set_title("Peak Nu")
    ax4.set_title("Mean Nu")
    ax1.set_xlabel("RG Step")
    ax2.set_xlabel("RG Step")
    ax3.set_xlabel("RG Step")
    ax4.set_xlabel("RG Step")

    x = []
    for i in range(start + 1, NUM_RG + 1):
        x.append(f"RG{i}")
    print(data["Peak Nu"].iloc[start:])
    ax1.scatter(x, data["Peak R2"].iloc[start:])
    ax2.scatter(x, data["Mean R2"].iloc[start:])
    ax3.scatter(x, data["Peak Nu"].iloc[start:])
    ax4.scatter(x, data["Mean Nu"].iloc[start:])
    calculate_average_nu(stats, 7, 10)
    plt.savefig(plots_filename, dpi=150)
    plt.close("Stats")
    print(f"Stats plotted and saved to {plots_filename}")
