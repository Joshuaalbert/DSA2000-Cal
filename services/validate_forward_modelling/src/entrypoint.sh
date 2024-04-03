#!/bin/bash

set -euo pipefail
set -x

wsclean --help

# Dirty image of visibilties

wsclean \
  -gridder idg \
  -idg-mode cpu \
  -wgridder-accuracy 1e-4 \
  -pol i \
  -name dirty_validation_image \
  -size 1024 1024 \
  -scale 1.6asec \
  -channels-out 1 \
  -nwlayers-factor 1 \
  -weight natural \
  -j 4 \
  /dsa/run/visibilities.ms
