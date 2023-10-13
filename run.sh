#!/bin/bash

set -e

# gets the script dir, should work with bash <script> and source <script>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Create run parameters: observation configuration, sky model
source "$SCRIPT_DIR"/run_prepare_run.sh

# Compute gains: Ionosphere simulation
source "$SCRIPT_DIR"/run_simulate_ionosphere.sh

# Compute gains: Instrumental effects
#echo source $"SCRIPT_DIR"/run_simulate_instrumental.sh

# Simulate visibilities: DFT
source "$SCRIPT_DIR"/run_predict_dft.sh
#
## Simulate visibilities: FFT
#echo source $"SCRIPT_DIR"/run_predict_fft.sh

# Simulate visibilities: RFI
source "$SCRIPT_DIR"/run_simulate_rfi.sh

# Sum visibilities and add noise
source "$SCRIPT_DIR"/run_sum_visibilities.sh

## Image: Create dirty image using WSClean
#echo source $"SCRIPT_DIR"/run_predict_image.sh
#
## Calibrate: prepare sky model
#echo source $"SCRIPT_DIR"/run_calibrate_prepare.sh
#
## Calibrate: run quartical and create subtracted visibilities
#echo source $"SCRIPT_DIR"/run_calibration.sh
#
## Image: image the subtracted visibilities
#echo source $"SCRIPT_DIR"/run_image.sh
