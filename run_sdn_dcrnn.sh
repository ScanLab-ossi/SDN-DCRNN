#!/bin/bash

set -e
set -x

exit_with_msg() {
  echo
  echo $1
  echo Exiting...
  exit 1
}

if [ -z "$EXP_DIR" ] ; then
  exit_with_msg "The variable EXP_DIR must be set to the name of the folder containing the experiment files!"
fi

if [ -z "$EXP_FILE" ] ; then
  exit_with_msg "The variable EXP_FILE must be set to the name of the file containing the experiment HD5!"
fi

if [ -z "$HORIZON" ] ; then
  exit_with_msg "The variable HORIZON must be set to the amount of datapoints to forecast into the future!"
fi

if [ -z "$SEQ_LEN" ] ; then
  exit_with_msg "The variable SEQ_LEN must be set to the amount of datapoints to be used as input!"
fi

# generate_training_data + HD5 ==> data npz
if [ -z "$PERIOD_CYCLE" ] ; then
  python scripts/generate_training_data.py --traffic_df_filename=$EXP_DIR/$EXP_FILE \
                                           --output_dir=$EXP_DIR \
                                           --horizon_len=$HORIZON
  INPUT_DIM=1

else
  python scripts/generate_training_data.py --traffic_df_filename=$EXP_DIR/$EXP_FILE \
                                           --output_dir=$EXP_DIR \
                                           --horizon_len=$HORIZON \
                                           --period-cycle-seconds=$PERIOD_CYCLE
  INPUT_DIM=2
fi
# gen_adj_mx + json ==> adj_mx.pkl + ports.num
NETWORK_JSON=$EXP_DIR/network_data.json
python scripts/gen_adj_mx.py --network-data-json=$NETWORK_JSON
# gen_config + paths + nodes num ==> config file
GRAPH_PKL=$EXP_DIR/adj_mx.pkl
PORT_NUM=$(cat $EXP_DIR/ports.num)
CONFIG_FILE=$EXP_DIR/sdn-dcrnn-config.yaml
python scripts/gen_config.py --dataset_dir=$EXP_DIR \
                             --graph_adj_mx_pkl=$GRAPH_PKL \
                             --num_ports=$PORT_NUM \
                             --horizon=$HORIZON \
                             --seq_len=$SEQ_LEN \
                             --input-dim=$INPUT_DIM \
                             --output-file=$CONFIG_FILE
# dcrnn_train + config file ==> model
python dcrnn_train.py --config_file=$CONFIG_FILE
# run_demo + trained config file ==> new predictions
PREDICTIONS_FILE=$EXP_DIR/predictions.npz
PREDICTIONS_CONFIG_FILE=`ls -1t $EXP_DIR/dcrnn*/config*.yaml | head -n 1`
python run_demo.py --config_file=$PREDICTIONS_CONFIG_FILE --output_filename=$PREDICTIONS_FILE
# plot_predictions + predictions ==> plots
PLOTS_DIR=$EXP_DIR/plots
mkdir -p $PLOTS_DIR
python scripts/plot_predictions.py --predictions-file=$PREDICTIONS_FILE --output-dir=$PLOTS_DIR
