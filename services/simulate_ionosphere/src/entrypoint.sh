#!/bin/bash

# Run from /dsa/input

set -euo pipefail
set -x

# Run Simulate Ionosphere

RUN_DIR=/dsa/run/sim_dsa2000W_1000m_grid_"$SPECIFICATION"_"$DURATION"_"$RESOLUTION"_"$SUFFIX"
mkdir -p $RUN_DIR

ln -sf $RUN_DIR /dsa/output                                                                             # for output simulation
OUTPUT_NAME=/dsa/output/sim_dsa2000W_1000m_grid_"$SPECIFICATION"_"$DURATION"_"$RESOLUTION"_"$SUFFIX".h5 # output h5parm

cd /dsa/output

cp /dsa/code/tomographic_kernel/bin/*.cfg .
cp /dsa/code/sky_model.bbs .

time python /dsa/code/tomographic_kernel/bin/simulate_ionosphere_phase_screen.py \
  --output_h5parm="$OUTPUT_NAME" \
  --phase_tracking='00h00m0.0s +37d07m47.400s' \
  --array_name=dsa2000W \
  --start_time='2019-03-19T19:58:14.9' \
  --time_resolution="$RESOLUTION" \
  --duration="$DURATION" \
  --field_of_view_diameter=4.0 \
  --avg_direction_spacing=10000.0 \
  --specification="$SPECIFICATION" \
  --ncpu=32 \
  --Nf=64 \
  --min_freq=700 \
  --max_freq=710.239 \
  --sky_model=sky_model.bbs

# Followup with interpolating onto DSA2000 array
