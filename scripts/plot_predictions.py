import numpy as np
import matplotlib.pylab as plt
from math import ceil
import argparse
from os.path import join as pj

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--predictions-file", type=str, default="DCRNN npz file"
)
parser.add_argument(
    "-o", "--output-dir", type=str, default="Output directory for PNG figures."
)

args = parser.parse_args()

ds = np.load(args.predictions_file)

horizon_len = ds['predictions'].shape[0]
num_nodes = ds['predictions'].shape[2]

predictions = ds['predictions'].transpose()
ground_truth = ds['groundtruth'].transpose()

for node in range(num_nodes):
    fig, axes = plt.subplots(nrows=ceil(horizon_len/4), ncols=4, sharex=True, sharey=True, figsize=(15, 15))
    node_predictions = predictions[node].transpose()
    node_ground_truth = ground_truth[node].transpose()
    axes_counter = 0
    for step_axes in axes.reshape(-1):
        step_axes.plot(node_predictions[axes_counter], label='predictions')
        step_axes.plot(node_ground_truth[axes_counter], label='ground truth')
        step_axes.set_title('Horizon distance = {}'.format(axes_counter+1))
        axes_counter += 1
    handles, labels = step_axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    fig.text(0.5, 0.04, "Sample Time", ha='center')
    fig.text(0.04, 0.5, "Prediction vs Truth", va='center', rotation='vertical')
    fig.savefig(pj(args.output_dir, "{0}-node-pred-vs-truth-plot.png".format(node)))
    plt.close(fig)

    node_ground_truth = ground_truth[node].transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(node_ground_truth[0], label='ground truth')
    fig.savefig(pj(args.output_dir, "{0}-node-ground-truth-plot.png".format(node)))
    plt.close(fig)


