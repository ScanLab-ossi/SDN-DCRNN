import argparse
import matplotlib.pylab as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--error-file", type=str, default="DCRNN final error rates file"
    )
    parser.add_argument("-n", "--network-name", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = {
        "MAE": [],
        "MAPE": [],
        "RMSE": []
    }
    horizons = 0
    with open(args.error_file) as file:
        for line in file:
            # example line: "Horizon 01, MAE: 0.03, MAPE: 0.3864, RMSE: 0.07"
            parts = line.split(", ")
            horizons += 1
            data["MAE"].append(float(parts[1].split(" ")[1]))
            data["MAPE"].append(float(parts[2].split(" ")[1]))
            data["RMSE"].append(float(parts[3].split(" ")[1]))
    print(data)
    figure = plt.figure(figsize=(60, 15))
    axes = figure.add_subplot(111)
    for error_type in data.keys():
        axes.plot(data[error_type], label=error_type)
    axes.set_title('Error Rates Over Prediction Horizon - {}'.format(args.network_name), fontsize=30)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, loc='upper left', fontsize=30)
    axes.set_xlabel("Prediction Horizon [sec]", fontsize=30)
    # axes.set_xticks(range(0, horizons, 50))
    # axes.set_xlim(0, horizons)
    axes.set_ylabel("Error Rates", fontsize=30)
    figure.savefig(args.network_name+"_error_rates.png", bbox_inches='tight', pad_inches=0)
    plt.close(figure)
