#!/bin/bash

PREFIX=/lustre/rbyrne/2024-03-03

python main.py \
  --data_ms="${PREFIX}/20240303_133205_73MHz.ms" \
  --subtract_ms_list="${PREFIX}/20240303_133205_73MHz_model_Cas.ms,${PREFIX}/20240303_133205_73MHz_model_Cyg.ms,${PREFIX}/20240303_133205_73MHz_model_Vir.ms" \
  --no_subtract_ms_list="${PREFIX}/20240303_133205_73MHz_model_diffuse.ms" \
  --times_per_chunk=1
