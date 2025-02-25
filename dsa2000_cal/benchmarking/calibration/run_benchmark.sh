#!/bin/bash

conde create -n cal_benchmark python=3.11
conda activate cal_benchmark

pip install jax[cuda12] jaxlib 'numpy<2'

python standalone_lm_multi_step.py

conda deactivate
conda remove -n cal_benchmark --all