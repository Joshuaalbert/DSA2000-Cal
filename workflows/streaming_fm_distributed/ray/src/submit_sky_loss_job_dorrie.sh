#!/bin/bash

echo "Submitting job on $(hostname) at $(date)"

cd /dsa/run && CUDA_VISIBLE_DEVICES=0 python /dsa/code/src/sky_loss/main_survey.py \
  --gpu_idx=0 \
  --node_idx=0 \
  --num_nodes=1

cd /dsa/run && CUDA_VISIBLE_DEVICES=0 python /dsa/code/src/sky_loss/main_vary_systematics.py \
  --gpu_idx=0 \
  --node_idx=0 \
  --num_nodes=1

echo "Job done at $(date)"
