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

if [ -z "$PARALLEL" ] ; then
	PARALLEL=4
	echo "The variable PARALLEL was not set, default=$PARALLEL will be used for parallelization"
else
	echo "The variable PARALLEL was set to $PARALLEL and will be used for parallelization"
fi

export ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")

ls -d $EXP_DATA_PATH/*/ | parallel -j $PARALLEL 'export EXP_DIR={} ; echo Running SDN-DCRNN for directory: $EXP_DIR ; $ROOT_DIR/run_sdn_dcrnn.sh'

