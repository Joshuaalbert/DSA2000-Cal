#!/bin/bash

set -euo pipefail
set -x

# This is what you need to call this in the DATA_DIR_HOST
export INPUT_MEASUREMENT_SET=/dsa/data/forward_model_ms # Should be in the data directory
export OUTPUT_MEASUREMENT_SET=visibilities.ms # Will be in the run directory (current)

time python3 /dsa/code/src/main.py
