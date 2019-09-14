#!/bin/bash

exit_with_msg() {
  echo
  echo $1
  echo Exiting...
  exit 1
}

if [ -z "$EXP_DATA_PATH" ] ; then
	  exit_with_msg "The variable EXP_DATA_PATH must be set to the storage path containing data directories for analysis!"
fi

ROOT_DIR=$(dirname "${BASH_SOURCE[0]}")

for EXP_DIR in `ls -d */` ; do
  echo Running SDN-DCRNN for directory: $EXP_DIR
  export EXP_DIR
  "$ROOT_DIR"/run_sdn_dcrnn.sh
done
