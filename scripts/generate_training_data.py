from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def make_sure_no_inf_nan(data):
    data_sum = np.sum(data)
    if np.isnan(data_sum):
        raise RuntimeError("data has NaN values")
    if np.isinf(data_sum):
        raise RuntimeError("data has Inf values")


def generate_graph_seq2seq_io_data(
        df, used_percentage, x_offsets, y_offsets, add_time_in_day=False
):
    """
    Generate samples from
    :param df:
    :param used_percentage:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    num_samples = int(num_samples * used_percentage)
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        # get the relative time in the day as a number in range [0,1]
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        # add as another dimension in each sample
        data_list.append(time_in_day)

    data = np.concatenate(data_list, axis=-1)
    try:
        make_sure_no_inf_nan(data)
    except RuntimeError:
        data = np.nan_to_num(data)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-(args.horizon_len-1), 1, args.horizon_step),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, args.horizon_len+1, args.horizon_step))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        args.used_sample_percentage,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # 0.7 is used for training, 0.1 is used for validation, 0.2 is used for testing
    num_samples = x.shape[0]
    num_test = int(num_samples * 0.2)
    num_train = int(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory."
    )
    parser.add_argument(
        "--add_time_in_day", action="store_true", help="Enable adding the time in day dimension to samples."
    )
    parser.add_argument(
        "--used-sample-percentage", type=float, default=1.0, help="Percentage of samples to use in analysis."
    )
    parser.add_argument(
        "--horizon_len", type=int, default=12, help="Length of the prediction horizon."
    )
    parser.add_argument(
        "--horizon_step", type=int, default=1, help="Length of the prediction horizon step."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        required=True,
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
