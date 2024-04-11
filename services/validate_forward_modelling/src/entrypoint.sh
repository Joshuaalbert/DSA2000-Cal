#!/bin/bash

set -euo pipefail
set -x


###
# Dirty image of visibilities

wsclean --help

wsclean \
  -gridder idg \
  -idg-mode cpu \
  -wgridder-accuracy 1e-4 \
  -pol i \
  -name dirty_validation_image \
  -size 6144 6144 \
  -scale 1.6asec \
  -channels-out 1 \
  -nwlayers-factor 1 \
  -weight natural \
  -j 4 \
  /dsa/run/visibilities.ms
