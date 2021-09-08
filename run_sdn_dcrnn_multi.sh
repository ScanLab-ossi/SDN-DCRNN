#!/bin/bash

set -xe

exit_with_msg() {
  echo
  echo $1
  echo Exiting...
  exit 1
}

if [ -z "$EXP_DATA_PATH" ] ; then
	exit_with_msg "The variable EXP_DATA_PATH must be set to the storage path containing data directories for analysis!"
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

PARALLEL_DEFAULT=2
if [ -z "$PARALLEL" ] ; then
	echo "The variable PARALLEL was not set, default=$PARALLEL_DEFAULT will be used for parallelization"
	PARALLEL=$PARALLEL_DEFAULT
else
	echo "The variable PARALLEL was set to $PARALLEL and will be used for parallelization"
fi

export ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")

ls -d $EXP_DATA_PATH/*/ | parallel -j $PARALLEL 'export EXP_DIR={} ; echo Running SDN-DCRNN for directory: $EXP_DIR ; $ROOT_DIR/run_sdn_dcrnn.sh'

