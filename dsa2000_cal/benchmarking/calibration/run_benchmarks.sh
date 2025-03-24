#!/bin/bash

conda create -n cal_benchmark python=3.11
conda activate cal_benchmark

pip install jax[cuda12] jaxlib 'numpy<2' nvtx

pip install ../..

python benchmark_R_calculation_TBC_gpu.py
python benchmark_JtR_calculation_TBC_gpu.py
python benchmark_JtJg_calculation_TBC_gpu.py

python benchmark_calibration_gpu.py

conda deactivate
conda remove -n cal_benchmark --all
