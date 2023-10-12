#!/bin/bash

# gets the script dir, should work with bash <script> and source <script>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Create run parameters: observation configuration, sky model
source "$SCRIPT_DIR"/run_prepare_run.sh

# Compute gains: Ionosphere simulation
echo source "$SCRIPT_DIR"/run_simulate_ionosphere.sh

# Compute gains: Instrumental effects
#echo source $"SCRIPT_DIR"/run_simulate_instrumental.sh
#
## Simulate visibilities: DFT
#echo source $"SCRIPT_DIR"/run_predict_dft.sh
#
## Simulate visibilities: FFT
#echo source $"SCRIPT_DIR"/run_predict_fft.sh
#
## Simulate visibilities: RFI
#echo source $"SCRIPT_DIR"/run_predict_rfi.sh
#
## Sum visibilities and add noise
#echo source $"SCRIPT_DIR"/run_predict_sum.sh
#
## Image: Create dirty image using WSClean
#echo source $"SCRIPT_DIR"/run_predict_image.sh
#
## Calibrate: prepare sky model
#echo source $"SCRIPT_DIR"/run_calibrate_prepare.sh
#
## Calibrate: run quartical and create subtracted visibilities
#echo source $"SCRIPT_DIR"/run_quartical.sh
#
## Image: image the subtracted visibilities
#echo source $"SCRIPT_DIR"/run_image.sh
