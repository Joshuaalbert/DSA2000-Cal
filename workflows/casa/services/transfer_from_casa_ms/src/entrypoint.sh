#!/bin/bash

set -euo pipefail
set -x

# This is what you need to call this in the DATA_DIR_HOST
export INPUT_CASA_MS=/dsa/data/lwa01.ms # Should be in the run directory
export OUTPUT_MEASUREMENT_SET=/dsa/run/lwa01_ms

time python3 /dsa/code/src/main.py
