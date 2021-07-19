import numpy as np
import matplotlib.pylab as plt
import argparse
import logging
from os.path import join as pj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--predictions-file", type=str, default="DCRNN npz file"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory for PNG figures."
    )
    parser.add_argument('--horizons', type=int, nargs='+', default=[7, 30],
                        help='Horizons to plot predictions for')
    parser.add_argument('-c', '--cut-off', type=int, default=3000,
                        help='Cut off point of the plot after this amount of samples, for better readability')
    return parser.parse_args()


def plot_multi_horizon_predictions_vs_ground_truth(horizons, predictions, ground_truth, output_path, cut_off):
    logging.debug("plot_max_horizon_predictions_vs_ground_truth")
    figure = plt.figure(figsize=(60, 15))
    axes = figure.add_subplot(111)
    plot_len = len(ground_truth[0])
    for horizon in horizons:
        # diff is created by the model in prediction calculation
        # used as start point to align all plots
        diff_to_max_horizon = max(horizons) - horizon
        # used to calculate end point to align all plots
        diff_to_min_horizon = horizon - min(horizons)
        # cut off end of plot if required
        end = min(cut_off, len(ground_truth[horizon-1]) - diff_to_min_horizon)
        plot_len = end - diff_to_max_horizon

        if horizon == min(horizons):
            # only need to plot ground truth once
            axes.plot(ground_truth[horizon-1][diff_to_max_horizon:end],
                      label='ground truth')
        axes.plot(predictions[horizon-1][diff_to_max_horizon:end],
                  label='prediction horizon {}'.format(horizon))
    axes.set_title('Horizon Predictions at {} vs Ground Truth'.format(horizons), fontsize=30)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, loc='upper center', fontsize=30)
    axes.set_xlabel("Sample Seconds", fontsize=30)
    axes.set_xticks(range(0, plot_len, 300))
    axes.set_xlim(0, plot_len)
    axes.set_ylabel("Prediction vs Truth", fontsize=30)
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    data_set = np.load(args.predictions_file)

    horizons_len = data_set['predictions'].shape[0]
    logging.info("Found %d horizons in data", horizons_len)
    if max(args.horizons) > horizons_len:
        logging.fatal("Requested horizons {} out of bound of horizons found in data {}"
                      .format(args.horizons, horizons_len))
    num_nodes = data_set['predictions'].shape[2]
    logging.info("Found %d nodes (ports) in data", num_nodes)

    predictions = data_set['predictions'].transpose()
    ground_truth = data_set['groundtruth'].transpose()

    for node in range(1):
        logging.info("Processing node #" + str(node))
        output_path = pj(args.output_dir, "predictions-vs-ground-truth-node-{}.png".format(node))
        node_ground_truth = ground_truth[node].transpose()
        node_predictions = predictions[node].transpose()
        plot_multi_horizon_predictions_vs_ground_truth(args.horizons,
                                                       node_predictions,
                                                       node_ground_truth,
                                                       output_path,
                                                       args.cut_off)

    logging.info("Completed all plots, saved to %s", args.output_dir)
