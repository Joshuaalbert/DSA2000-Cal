#!/bin/bash

set -euo pipefail
set -x

# This is what you need to call this in the DATA_DIR_HOST
export MEASUREMENT_SET=/dsa/run/forward_model_ms # Should be in the data directory

time python3 /dsa/code/src/main.py
