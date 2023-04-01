#!/bin/bash

set -euo pipefail

SKY_MODEL_BBS=skymodel.bbs
SKY_MODEL_TIGGER=skymodel.lsm.html

# Check if SKY_MODEL_TIGGER doesn't exist
if [[ ! -f "$SKY_MODEL_TIGGER" ]]; then

  # Check if SKY_MODEL_BBS is empty
  if [[ -z "$SKY_MODEL_BBS" ]]; then
    echo "$SKY_MODEL_BBS is empty. Aborting conversion."
    exit 1
  fi

  # Check if SKY_MODEL_BBS already exists
  if [[ ! -f "$SKY_MODEL_BBS" ]]; then
    echo "$SKY_MODEL_BBS doesn't exist. Aborting conversion."
    exit 1
  fi

  # Convert SKY_MODEL_BBS to SKY_MODEL_TIGGER
  tigger-convert -t BBS -o Tigger $SKY_MODEL_BBS $SKY_MODEL_TIGGER
fi

# Run Quartical

goquartical parset.yaml
