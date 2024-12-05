#!/bin/bash

set -euo pipefail
set -x


###
# Dirty image of visibilities

wsclean --help

# You probably should use a horizon-mask
#So: wsclean -multiscale -multiscale-scale-bias 0.8 -pol IV -size 4096 4096 -scale 0.03125 -niter 10000 -mgain 0.85 -weight briggs 0 -horizon-mask 5deg -no-update-model-required -name testv1 name.ms

wsclean \
  -gridder idg \
  -idg-mode cpu \
  -wgridder-accuracy 1e-4 \
  -multiscale -multiscale-scale-bias 0.8 \
  -pol IQUV \
  -link-polarizations I \
  -size 4096 4096 \
  -scale 0.03125 \
  -niter 10000 \
  -mgain 0.85 \
  -weight briggs 0 \
  -horizon-mask 5deg \
  -no-update-model-required \
  -channels-out 1 \
  -nwlayers-factor 1 \
  -j 4 \
  -continue \
  -name lwa01_full_sky \
  /dsa/data/lwa01.ms
