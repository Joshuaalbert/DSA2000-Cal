#!/bin/bash

set -euo pipefail
set -x

# This is what you need to call this in the DATA_DIR_HOST
export RUN_CONFIG=prepare_run_config.json

# Copy data from data dir to run dir
rsync -avP /dsa/data/ /dsa/run/

time python3 /dsa/code/src/main.py
