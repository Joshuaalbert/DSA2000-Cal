#!/bin/bash

# Assert that SKY_MODEL_TIGGER is a valid filename
if [[ ! -f $SKY_MODEL_TIGGER ]]; then
  echo "SKY_MODEL_TIGGER is not a valid filename"
  exit 1
fi

# Assert that SKY_MODEL_BBS is a valid filename
if [[ ! -f $SKY_MODEL_BBS ]]; then
  echo "SKY_MODEL_BBS is not a valid filename"
  exit 1
fi

# Check if SKY_MODEL_TIGGER doesn't exist
if [[ ! -f $SKY_MODEL_TIGGER ]]; then

  # Check if SKY_MODEL_BBS already exists
  if [[ -f $SKY_MODEL_BBS ]]; then
    echo "SKY_MODEL_BBS already exists. Aborting conversion."
    exit 1
  fi

  # Convert SKY_MODEL_BBS to SKY_MODEL_TIGGER
  tigger-convert -t BBS -o Tigger $SKY_MODEL_BBS $SKY_MODEL_TIGGER
fi

# Run Quartical

goquartical $PARSET