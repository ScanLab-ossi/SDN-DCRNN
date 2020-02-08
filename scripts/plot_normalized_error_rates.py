#!/usr/bin/env python

# plot_normalized_error_rates.py
# This script will plot normalized error rates for datasets
#
#################################################################################
import argparse
import logging
import matplotlib.pyplot as plt
from os.path import dirname, sep
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-hdf", required=True,
                        help="Input file name - a HDF file describing the normalized error rates")
    parser.add_argument("-o", "--output-path", default="",
                        help="Path to output plots, defaults to '<input-file-base-path>'.")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Set to add debug level log")
    args = parser.parse_args()
    if args.output_path == "":
        args.output_path = dirname(args.input_hdf) + sep
    return args


def save_error_to_experiment_figures(error_rates, experiments, horizons, output_path, file_format):
    figure_size = (40, 10)
    logging.info("Creating plots for error to experiment")
    for error_measure in ['RMSE', 'MAE', 'MAPE']:
        logging.info("Creating plots for error measure %s", error_measure)
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].title.set_text(error_measure + "/port_means_iqr")
        axes[1].title.set_text(error_measure + "/second_means_iqr")
        for horizon in horizons:
            group = error_rates.groupby(level=1).get_group(horizon)
            group[error_measure + "/port_means_iqr"] \
                .plot(ax=axes[0], figsize=figure_size)
            group[error_measure + "/second_means_iqr"] \
                .plot(ax=axes[1], figsize=figure_size)
        axes[0].legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        axes[1].legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        plt.xticks(range(len(experiments)), experiments, rotation=90)
        filename = output_path + file_format.format(error_measure)
        logging.info("Saving figure to %s", filename)
        plt.subplots_adjust(bottom=0.25, left=0.05)
        fig.savefig(filename)


def save_error_to_horizon_figures(error_rates, horizons, output_path, file_format):
    figure_size = (40, 10)
    logging.info("Creating plots for error to horizon")
    for error_measure in ['RMSE', 'MAE', 'MAPE']:
        logging.info("Creating plots for error measure %s", error_measure)
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].title.set_text(error_measure + "/port_means_iqr")
        axes[1].title.set_text(error_measure + "/second_means_iqr")
        for name, group in error_rates.groupby(level=0):
            group[error_measure + "/port_means_iqr"] \
                .plot(label=name[0], ax=axes[0], figsize=figure_size)
            group[error_measure + "/second_means_iqr"] \
                .plot(label=name[0], ax=axes[1], figsize=figure_size)
        axes[0].legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        axes[1].legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        plt.xticks(range(0, len(horizons)), horizons, rotation=90)
        filename = output_path + file_format.format(error_measure)
        logging.info("Saving figure to %s", filename)
        plt.subplots_adjust(bottom=0.25, left=0.05)
        fig.savefig(filename)


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    error_rates_df = pd.read_hdf(args.input_hdf, key='normalized_error_rates')
    experiment_list = [key[0] for key in list(error_rates_df.index.levels[0])]
    horizon_list = [name for (name, _) in error_rates_df.groupby(level=1)]
    save_error_to_experiment_figures(error_rates_df, experiment_list, horizon_list,
                                     args.output_path, "Normalized-{}-to-experiment.png")

    save_error_to_horizon_figures(error_rates_df, horizon_list, args.output_path, "Normalized-{}-to-horizon.png")
