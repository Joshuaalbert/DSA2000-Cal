#!/bin/bash

# Run from /dsa/run

set -euo pipefail
set -x

# Create output directory
mkdir -p calibration

time python3 /dsa/code/src/main.py
