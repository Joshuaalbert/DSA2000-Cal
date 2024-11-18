#!/bin/bash

echo "Submitting job on $(hostname) at $(date)"

cd /dsa/run && python /dsa/code/src/main.py \
  --array_name='dsa2000_31b' \
  --field_of_view=3. \
  --oversample_factor=5. \
  --full_stokes=True \
  --num_cal_facets=1 \
  --root_folder='/dsa/run/working_dir' \
  --run_name='demo' \

echo "Job done at $(date)"
