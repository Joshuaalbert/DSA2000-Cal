#!/bin/bash

set -euo pipefail

# Check if SKY_MODEL_TIGGER doesn't exist
if [[ ! -f "$SKY_MODEL_TIGGER" ]]; then

  # Check if SKY_MODEL_BBS is empty
  if [[ -z "$SKY_MODEL_BBS" ]]; then
    echo "SKY_MODEL_BBS is empty. Aborting conversion."
    exit 1
  fi

  echo "$SKY_MODEL_TIGGER doesn't exist. Converting from $SKY_MODEL_BBS."

  # Check if SKY_MODEL_BBS already exists
  if [[ -f "$SKY_MODEL_BBS" ]]; then
    echo "SKY_MODEL_BBS already exists. Aborting conversion."
    exit 1
  fi

  # Convert SKY_MODEL_BBS to SKY_MODEL_TIGGER
  tigger-convert -t BBS -o Tigger $SKY_MODEL_BBS $SKY_MODEL_TIGGER
fi

# Run Quartical

goquartical $PARSET
