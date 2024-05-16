#!/bin/bash

set -euo pipefail
set -x

# This is what you need to call this in the DATA_DIR_HOST
export MEASUREMENT_SET=/dsa/run/lwa01_ms # Will be in the run directory (current)

time python3 /dsa/code/src/main.py
