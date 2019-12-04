#!/usr/bin/env python

# calculate_normalized_error_rates.py
# This script will calculate normalized error rates for datasets by using its
# IQR (interquartile range).
#
#################################################################################
import argparse
import logging
from os.path import join as pj
from glob import glob
import json
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base-path", required=True,
                        help="Base directory containing dataset subdirectories")
    parser.add_argument("-i", "--iqr-json", default="sflow-datagrams.hd5-iqr.json",
                        help="Input file name - a JSON file describing the IQR")
    parser.add_argument("-e", "--error-rates", default="final_training_error_rates.txt",
                        help="Input file name - a TXT file describing the error rates found during training")
    parser.add_argument("-o", "--output-path", default="",
                        help="Path to output normalized data, defaults to '<data-base-path>/normalized.hd5'.")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Set to add debug level log")
    args = parser.parse_args()
    if args.output_path == "":
        args.output_path = pj(args.data_base_path, 'normalized.hd5')
    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    error_rates = {}
    for d in glob(pj(args.data_base_path, "*", "")):
        print(d)
        with open(pj(d, args.iqr_json)) as f:
            iqr_stats = json.load(f)
        dir_error_rates = {}
        with open(pj(d, args.error_rates)) as f:
            for line in f.readlines():
                split_line = line.strip().split(', ')
                horizon_numeric = int(split_line[0].split(" ")[1])
                mae = split_line[1].split(": ")[1]
                mape = split_line[2].split(": ")[1]
                rmse = split_line[3].split(": ")[1]
                dir_error_rates[split_line[0]] = {'Horizon#': horizon_numeric,
                                                  'MAE': float(mae),
                                                  'MAE/total_iqr':
                                                  float(mae) / float(iqr_stats["total_iqr"]["result"]),
                                                  'MAE/second_means_iqr':
                                                  float(mae) / float(iqr_stats["second_means_iqr"]["result"]),
                                                  'MAE/port_means_iqr':
                                                  float(mae) / float(iqr_stats["port_means_iqr"]["result"]),
                                                  'MAPE': float(mape),
                                                  'MAPE/total_iqr':
                                                  float(mape) / float(iqr_stats["total_iqr"]["result"]),
                                                  'MAPE/second_means_iqr':
                                                  float(mape) / float(iqr_stats["second_means_iqr"]["result"]),
                                                  'MAPE/port_means_iqr':
                                                  float(mape) / float(iqr_stats["port_means_iqr"]["result"]),
                                                  'RMSE': float(rmse),
                                                  'RMSE/total_iqr':
                                                  float(rmse) / float(iqr_stats["total_iqr"]["result"]),
                                                  'RMSE/second_means_iqr':
                                                  float(rmse) / float(iqr_stats["second_means_iqr"]["result"]),
                                                  'RMSE/port_means_iqr':
                                                  float(rmse) / float(iqr_stats["port_means_iqr"]["result"])}
        error_rates[d] = dir_error_rates

    error_rates_df = pd.DataFrame.from_dict({(i, j): error_rates[i][j]
                                            for i in error_rates.keys()
                                            for j in error_rates[i].keys()},
                                            orient='index')
    logging.info("Dumping results to " + args.output_path)
    logging.debug("JSON will be created from: %s", str(error_rates_df))
    error_rates_df.to_hdf(args.output_path, key='normalized_error_rates')
