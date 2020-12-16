import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from math import ceil
import argparse
import logging
from os.path import join as pj
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--predictions-file", type=str, help="DCRNN npz file"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory for PNG figures."
    )
    parser.add_argument(
        "-t", "--thresholds", type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6], help="Thresholds list."
    )
    parser.add_argument(
        "-d", "--debug", action='store_true', help="Set to add debug level log"
    )
    return parser.parse_args()


def plot_ground_truth(node_ground_truth_data, output_path):
    logging.debug("plot_ground_truth")
    figure = plt.figure(figsize=(60, 15))
    axes = figure.add_subplot(111)
    axes.plot(node_ground_truth_data, label='ground truth')
    axes.set_xticks(range(0, len(ground_truth[0]), 50))
    axes.set_xlim(0, len(ground_truth[0]))
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def plot_predictions_vs_ground_truth(nplots, predictions, ground_truth, output_path):
    logging.debug("plot_predictions_vs_ground_truth")
    figure, axes = plt.subplots(nrows=ceil(nplots / 4), ncols=4, sharex='all', sharey='all', figsize=(15, 15))
    for i, ax in enumerate(figure.axes):
        ax.plot(predictions[i], label='predictions')
        ax.plot(ground_truth[i], label='ground truth')
        ax.set_title('Horizon distance = {}'.format(i))
        ax.set_xticks(range(0, len(ground_truth[0]), 50))
        ax.set_xlim(0, len(ground_truth[0]))
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', fontsize=30)
    figure.text(0.5, 0.04, "Sample Seconds", ha="center", va="center", fontsize=30)
    figure.text(0.04, 0.5, "Prediction vs Truth", ha="center", va="center", rotation=90, fontsize=30)
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def plot_max_horizon_predictions_vs_ground_truth(max_horizon, predictions, ground_truth, output_path):
    logging.debug("plot_max_horizon_predictions_vs_ground_truth")
    figure = plt.figure(figsize=(60, 15))
    axes = figure.add_subplot(111)
    axes.plot(predictions[max_horizon-1], label='predictions')
    axes.plot(ground_truth[max_horizon-1], label='ground truth')
    axes.set_title('Horizon distance = {}'.format(max_horizon), fontsize=30)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, loc='upper center', fontsize=30)
    axes.set_xlabel("Sample Seconds", fontsize=30)
    axes.set_xticks(range(0, len(ground_truth[max_horizon-1]), 50))
    axes.set_xlim(0, len(ground_truth[max_horizon-1]))
    axes.set_ylabel("Prediction vs Truth", fontsize=30)
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def count_exact_passing_threshold_incidents(horizons, thresholds, node_ground_truth, node_predictions, template):
    logging.debug("count_exact_passing_threshold_incidents")
    passing_threshold_incidents = deepcopy(template)
    max_ground_truth = max(node_ground_truth[0])
    prev_data_point = 0
    data_counter = 0
    for data_point in node_ground_truth[0]:
        for threshold in thresholds:
            threshold_ground_truth = max_ground_truth * threshold
            if data_point >= threshold_ground_truth > prev_data_point:
                # threshold passed in truth, check if prediction passed too per horizon
                for horizon in horizons:
                    predicted_data_point = node_predictions[horizon - 1][data_counter + 1]
                    prev_predicted_data_point = node_predictions[horizon - 1][data_counter]
                    if predicted_data_point >= threshold_ground_truth > prev_predicted_data_point:
                        passing_threshold_incidents[horizon][threshold]["true"] += 1
                    else:
                        passing_threshold_incidents[horizon][threshold]["false"] += 1
        prev_data_point = data_point
        data_counter += 1
    return passing_threshold_incidents


def count_relaxed_passing_threshold_incidents(horizons, thresholds, node_ground_truths, node_predictions, template):
    logging.debug("count_relaxed_passing_threshold_incidents")
    passing_threshold_incidents = deepcopy(template)
    for horizon in horizons:
        max_ground_truth = max(node_ground_truths[horizon-1])
        prev_data_point = 0
        data_counter = 0
        for data_point in node_ground_truths[horizon - 1]:
            for threshold in thresholds:
                threshold_ground_truth = max_ground_truth * threshold
                if data_point >= threshold_ground_truth > prev_data_point:
                    delta = ceil(horizon/2)
                    half_delta = ceil(horizon/4)
                    # threshold passed in truth, check if prediction passed too
                    predicted_data_point = node_predictions[horizon-1][data_counter + 1 + delta]
                    prev_predicted_data_point = node_predictions[horizon-1][data_counter - half_delta]
                    if predicted_data_point >= threshold_ground_truth > prev_predicted_data_point:
                        passing_threshold_incidents[horizon][threshold]["true"] += 1
                    else:
                        passing_threshold_incidents[horizon][threshold]["false"] += 1
            prev_data_point = data_point
            data_counter += 1
    return passing_threshold_incidents


def plot_passing_threshold_recall(thresholds, horizons, horizon_prediction_results_per_node, node_keys, output_path):
    logging.debug("plot_passing_threshold_recall")
    logging.info("Calculating and plotting recall, using the following node keys: " + str(node_keys))
    figure = plt.figure()
    axes = figure.add_subplot(211)
    extracted_horizons = [5, 10, 20, 30, 60]
    recall_table = pd.DataFrame(index=thresholds, columns=extracted_horizons)
    for threshold in thresholds:
        recalls = []
        for horizon in horizons:
            true = sum([horizon_prediction_results_per_node[n][horizon][threshold]["true"] for n in node_keys])
            false = sum([horizon_prediction_results_per_node[n][horizon][threshold]["false"] for n in node_keys])
            if true + false > 0:
                recall = true / (true + false)
            else:
                recall = 0
            recalls.append(recall)
        axes.plot(recalls, label="Threshold = " + str(threshold) + ' Samples = ' + str(true + false))
        recall_table.loc[threshold] = [recalls[i-1] for i in extracted_horizons]
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, loc='upper center')
    axes.set_xlabel("Horizons")
    axes.set_xticks(range(0, len(horizons), 5))
    axes.set_xlim(0, len(horizons))
    axes.set_ylabel("Thresholds Recall")
    logging.info("Recalls:\nThresholds \\ horizons\n" + str(recall_table))
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def get_top_third_busy_nodes(node_sums):
    logging.debug("get_top_third_busy_nodes")
    sorted_node_sums = sorted(node_sums.items(), key=lambda x: x[1], reverse=True)
    top = ceil(len(node_sums.keys())/3)
    return [node[0] for node in sorted_node_sums[:top]]


if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    data_set = np.load(args.predictions_file)

    horizons_len = data_set['predictions'].shape[0]
    logging.info("Will produce %d horizons", horizons_len)
    num_nodes = data_set['predictions'].shape[2]
    logging.info("Will produce results for %d nodes", num_nodes)

    predictions = data_set['predictions'].transpose()
    ground_truth = data_set['groundtruth'].transpose()

    threshold_map = dict((t, {"true": 0, "false": 0}) for t in args.thresholds)
    horizons = [h for h in range(1, horizons_len + 1)]
    passing_threshold_incidents_template = dict((el, deepcopy(threshold_map)) for el in horizons)
    exact_passing_threshold_incidents_per_node = {}
    relaxed_passing_threshold_incidents_per_node = {}
    node_sums = {}
    for node in range(num_nodes):
        logging.info("Processing node #%d", node)
        node_ground_truth = ground_truth[node].transpose()
        node_sums[node] = node_ground_truth.sum()

        logging.debug("Here you can see if all horizons have the same ground truth:")
        for h in horizons:
            logging.debug(node_ground_truth[h-1][61-h:121-h])

        plot_ground_truth(node_ground_truth[0],
                          pj(args.output_dir, "{0}-node-ground-truth-plot.png".format(node)))

        node_predictions = predictions[node].transpose()
        plot_predictions_vs_ground_truth(horizons_len,
                                         node_predictions,
                                         node_ground_truth,
                                         pj(args.output_dir,
                                            "{0}-node-prediction-vs-truth-plot.png".format(node)))

        plot_max_horizon_predictions_vs_ground_truth(horizons_len,
                                                     node_predictions,
                                                     node_ground_truth,
                                                     pj(args.output_dir,
                                                        "{0}-node-max-horizon-prediction-vs-truth-plot.png"
                                                        .format(node)))

        exact_passing_threshold_incidents_per_node[node] = count_exact_passing_threshold_incidents(
                                                           horizons,
                                                           args.thresholds,
                                                           node_ground_truth,
                                                           node_predictions,
                                                           passing_threshold_incidents_template)

        relaxed_passing_threshold_incidents_per_node[node] = count_relaxed_passing_threshold_incidents(
                                                             horizons,
                                                             args.thresholds,
                                                             node_ground_truth,
                                                             node_predictions,
                                                             passing_threshold_incidents_template)

    plot_passing_threshold_recall(args.thresholds,
                                  horizons,
                                  exact_passing_threshold_incidents_per_node,
                                  node_sums.keys(),
                                  pj(args.output_dir, "all-nodes-exact-thresholds-recall-plot.png"))

    plot_passing_threshold_recall(args.thresholds,
                                  horizons,
                                  relaxed_passing_threshold_incidents_per_node,
                                  node_sums.keys(),
                                  pj(args.output_dir, "all-nodes-relaxed-thresholds-recall-plot.png"))

    top_third_busy_nodes = get_top_third_busy_nodes(node_sums)
    plot_passing_threshold_recall(args.thresholds,
                                  horizons,
                                  exact_passing_threshold_incidents_per_node,
                                  top_third_busy_nodes,
                                  pj(args.output_dir, "top-third-busy-nodes-exact-thresholds-recall-plot.png"))

    plot_passing_threshold_recall(args.thresholds,
                                  horizons,
                                  relaxed_passing_threshold_incidents_per_node,
                                  top_third_busy_nodes,
                                  pj(args.output_dir, "top-third-busy-nodes-relaxed-thresholds-recall-plot.png"))
