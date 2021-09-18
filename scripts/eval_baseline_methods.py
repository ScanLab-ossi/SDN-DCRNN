import argparse
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from lib.utils import StandardScaler

import matplotlib.pylab as plt


def exp_smoothing_predict(df, n_forwards=(1, 3), test_ratio=0.2):
    """
    Multivariate time series forecasting using an ExponentialSmoothing Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    # Do forecasting.
    result = np.zeros(shape=(n_test, n_output))
    for i in range(n_output):
        exp_s_model = ExponentialSmoothing(df_train.iloc[i].values)
        exp_s_result = exp_s_model.fit()
        result[:, i] = exp_s_result.forecast(n_test)

    df_predicts = result * len(n_forwards)

    return df_predicts, df_test


def eval_exp_smoothing(traffic_reading_df, horizons):
    y_predicts, y_test = exp_smoothing_predict(traffic_reading_df, n_forwards=horizons, test_ratio=0.2)
    logger.info('Exp. Smoothing')
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(horizons):
        rmse = masked_rmse_np(preds=y_predicts[i].values(), labels=y_test.values(), null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].values(), labels=y_test.values(), null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values(), labels=y_test.values(), null_val=0)
        line = 'Exp. Smoothing\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)
    plot_eval_first_node_at_max_horizon("ExponentialSmoothing", horizons, y_predicts, y_test)


def var_predict(df, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    np.nan_to_num(df_train, copy=False)
    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test


def eval_var(traffic_reading_df, horizons, n_lags=3):
    y_predicts, y_test = var_predict(traffic_reading_df, n_forwards=horizons, n_lags=n_lags,
                                     test_ratio=0.2)
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(horizons):
        rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)
    plot_eval_first_node_at_max_horizon("VAR", horizons, y_predicts, y_test)


def plot_eval_first_node_at_max_horizon(eval_method_name, horizons, y_predicts, y_test):
    figure = plt.figure(figsize=(60, 15))
    axes = figure.add_subplot(111)
    axes.plot(y_predicts[max(horizons)].iloc[0], label=f"{eval_method_name} prediction max horizon")
    axes.plot(y_test.iloc[0], label="Real Data")
    axes.set_title(f'{eval_method_name} Prediction at max horizon', fontsize=30)
    axes.set_xlabel("Prediction Time [sec]", fontsize=30)
    axes.set_ylabel("Predicted vs Real Rates", fontsize=30)
    figure.savefig(f"{eval_method_name}_prediction.png", bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def main(args):
    traffic_reading_df = pd.read_hdf(args.traffic_reading_filename)
    assert isinstance(traffic_reading_df, DataFrame)
    eval_exp_smoothing(traffic_reading_df, args.horizons)
    eval_var(traffic_reading_df, args.horizons, n_lags=3)


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', type=str, required=True,
                        help='Path to the traffic Dataframe.')
    parser.add_argument('--horizons', type=int, nargs='+', default=range(1, 30),
                        help='Horizons to evaluate for.')
    args = parser.parse_args()
    main(args)
