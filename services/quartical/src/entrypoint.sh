#!/bin/bash

# Run from /dsa/input

set -euo pipefail
set -x

ln -s $PWD/$RUN_NAME $PWD/input

## Prepare run dir

MS_NAME=$(basename "$MS_NAME")
RUN_NAME_MS=$(echo "$MS_NAME" | rev | cut -f2- -d'.' | rev)
RUN_DIR=/dsa/run/$RUN_NAME_MS/$RUN_NAME
mkdir -p $RUN_DIR

# Links to standard names for parset.yaml to use
ln -sf $RUN_DIR /dsa/output                    # for output solutions and logs
ln -sf /dsa/data/$MS_NAME /dsa/output/input.ms # for ms input

# for clarity put sky models in output
cp input/parset.yaml /dsa/output

## These should match input_model.recipe in the parset.yaml
SKY_MODEL_BBS_FILES="input/skymodel*.bbs"

for SKY_MODEL_BBS in $SKY_MODEL_BBS_FILES; do
  # Replace the file .bbs suffix with .lsm.html
  SKY_MODEL_TIGGER="${SKY_MODEL_BBS%.bbs}.lsm.html"

  # store it in run dir for posterity
  cp $SKY_MODEL_TIGGER /dsa/output

  # Run the convert script
  convert-script "$SKY_MODEL_BBS" "$SKY_MODEL_TIGGER"
done

# Run Quartical

cd /dsa/output

time goquartical parset.yaml
