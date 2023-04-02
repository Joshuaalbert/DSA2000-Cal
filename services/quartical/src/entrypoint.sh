#!/bin/bash

# Run from /dsa/input

set -euo pipefail

## These should match parset.yaml
SKY_MODEL_BBS=skymodel.bbs
SKY_MODEL_TIGGER=skymodel.lsm.html

# Check if SKY_MODEL_TIGGER doesn't exist
if [[ ! -f "$SKY_MODEL_TIGGER" ]]; then

  # Check if SKY_MODEL_BBS exists
  if [[ ! -f "$SKY_MODEL_BBS" ]]; then
    echo "$SKY_MODEL_BBS doesn't exist. Aborting conversion."
    exit 1
  fi

  # Convert SKY_MODEL_BBS to SKY_MODEL_TIGGER
  tigger-convert -t BBS -o Tigger $SKY_MODEL_BBS $SKY_MODEL_TIGGER
fi

# Run Quartical

MS_NAME=$(basename "$MS_NAME")
RUN_NAME=$(echo "$MS_NAME" | rev | cut -f2- -d'.' | rev)
RUN_DIR=/dsa/run/$RUN_NAME
mkdir $RUN_DIR
mkdir $RUN_DIR/gains
mkdir $RUN_DIR/logs

# Links to standard names for parset.yaml to use
ln -s /dsa/data/$MS_NAME /dsa/run/input.ms
ln -s $RUN_DIR/gains /dsa/run/gains
ln -s $RUN_DIR/logs /dsa/run/logs

export MS_NAME
export RUN_NAME
export RUN_DIR

goquartical parset.yaml
