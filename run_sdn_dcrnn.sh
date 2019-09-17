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

# generate_training_data + HD5 ==> data npz
python scripts/generate_training_data.py --traffic_df_filename=$EXP_DIR/sflow-datagrams.hd5 --output_dir=$EXP_DIR
# gen_adj_mx + csv ==> adj mx pkl
LINKS_CSV=`ls $EXP_DIR/*.graphml-topo.csv`
python scripts/gen_adj_mx.py --links_csv=$LINKS_CSV  --intfs_list=$EXP_DIR/intfs-list
# gen_config + paths + nodes num ==> config file
GRAPH_PKL=`ls $EXP_DIR/*.pkl`
PORT_NUM=`cat $EXP_DIR/*.port.num`
python scripts/gen_config.py --dataset_dir=$EXP_DIR --graph_adj_mx_pkl=$GRAPH_PKL --num_ports=$PORT_NUM
# dcrnn_train + config file ==> model
CONFIG_FILE=`ls $EXP_DIR/*.yaml`
python dcrnn_train.py --config_file=$CONFIG_FILE
# run_demo + trained config file ==> new predictions
PREDICTIONS_FILE=$EXP_DIR/predictions.npz
PREDICTIONS_CONFIG_FILE=`ls -1t $EXP_DIR/dcrnn*/config*.yaml | head -n 1`
python run_demo.py --config_file=$PREDICTIONS_CONFIG_FILE --output_filename=$PREDICTIONS_FILE
# plot_predictions + predictions ==> plots
PLOTS_DIR=$EXP_DIR/plots
mkdir $PLOTS_DIR
python scripts/plot_predictions.py --predictions-file=$PREDICTIONS_FILE --output-dir=$PLOTS_DIR
