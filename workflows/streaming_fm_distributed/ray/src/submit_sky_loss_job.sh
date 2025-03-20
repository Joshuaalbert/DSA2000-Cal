#!/bin/bash

echo "Submitting job on $(hostname) at $(date)"

node_idx=0

for gpu_idx in $(seq 0 9); do

  cd /dsa/run && python /dsa/code/src/sky_loss/main_survey.py \
    --gpu_idx=$gpu_idx \
    --node_idx=$node_idx \
    --num_nodes=2 &

done

for gpu_idx in $(seq 0 9); do

  cd /dsa/run && python /dsa/code/src/sky_loss/main_vary_systematics.py \
    --gpu_idx=$gpu_idx \
    --node_idx=$node_idx \
    --num_nodes=2 &

done

echo "Job done at $(date)"
