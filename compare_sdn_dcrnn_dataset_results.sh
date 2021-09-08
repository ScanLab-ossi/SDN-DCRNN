#!/bin/bash

set -e

exit_with_msg() {
  echo
  echo $1
  echo Exiting...
  exit 1
}

if [ -z "$EXP_DATA_PATH" ] ; then
	  exit_with_msg "The variable EXP_DATA_PATH must be set to the storage path containing data directories for analysis!"
fi

if [ -z "$HORIZON" ] ; then
  exit_with_msg "The variable HORIZON must be set to the amount of datapoints to forecast into the future!"
fi

ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")

for EXP_DIR in `ls -d $EXP_DATA_PATH/*/` ; do
  echo Gathering SDN-DCRNN model errors for directory: $EXP_DIR
  tail -n$HORIZON $EXP_DIR/dcrnn*/info.log | cut -d" " -f8- > $EXP_DIR/final_training_error_rates.txt
done

python $ROOT_DIR/scripts/calculate_normalized_error_rates.py --data-base-path=$EXP_DATA_PATH

python $ROOT_DIR/scripts/plot_normalized_error_rates.py -i=$EXP_DATA_PATH/normalized.h5
