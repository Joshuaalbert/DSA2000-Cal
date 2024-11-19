#!/bin/bash

# copy local with rsync -avP
PREFIX=/lustre/rbyrne/2024-03-03
rsync -avP ${PREFIX}/20240303_133205_73MHz.ms ${LOCAL_PREFIX}/20240303_133205_73MHz.ms
rsync -avP ${PREFIX}/20240303_133205_73MHz_model_Cas.ms ${LOCAL_PREFIX}/20240303_133205_73MHz_model_Cas.ms
rsync -avP ${PREFIX}/20240303_133205_73MHz_model_Cyg.ms ${LOCAL_PREFIX}/20240303_133205_73MHz_model_Cyg.ms
rsync -avP ${PREFIX}/20240303_133205_73MHz_model_Vir.ms ${LOCAL_PREFIX}/20240303_133205_73MHz_model_Vir.ms
rsync -avP ${PREFIX}/20240303_133205_73MHz_model_diffuse.ms ${LOCAL_PREFIX}/20240303_133205_73MHz_model_diffuse.ms

LOCAL_PREFIX=$PWD

python main.py \
  --data_ms="${LOCAL_PREFIX}/20240303_133205_73MHz.ms" \
  --subtract_ms_list="${LOCAL_PREFIX}/20240303_133205_73MHz_model_Cas.ms,${LOCAL_PREFIX}/20240303_133205_73MHz_model_Cyg.ms,${LOCAL_PREFIX}/20240303_133205_73MHz_model_Vir.ms" \
  --no_subtract_ms_list="${LOCAL_PREFIX}/20240303_133205_73MHz_model_diffuse.ms" \
  --times_per_chunk=1

