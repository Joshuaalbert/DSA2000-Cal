#!/bin/bash

set -euo pipefail
set -x

declare -A services

# The values represent whether to use --no-cache or not. 1 for yes, 0 for no.
services=(
  #  ["prepare_run"]=1
  #  ["simulate_ionosphere"]=0
  #  ["simulate_instrumental_effects"]=0
  #  ["predict_dft"]=1
  #  ["predict_fft"]=0
  ["simulate_rfi"]=0
  #  ["sum_visibilities"]=1
  #  ["dirty_image"]=0
  #  ["calibration"]=0
  #  ["image_subtracted"]=0
  #  ["image_a_proj"]=0
)

run_services() {
  local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

  for service_name in "${!services[@]}"; do
    echo "Processing service: $service_name"

    local cache_option=""
    if [ ${services["$service_name"]} -eq 1 ]; then
      cache_option="--no-cache"
    fi

    # Force rebuild considering the cache option
    docker compose -f "$script_dir"/docker-compose.yaml build $cache_option "$service_name"
    docker compose -f "$script_dir"/docker-compose.yaml up -d --force-recreate "$service_name"

    # Stream the logs until the service ends.
    docker compose logs -f "$service_name"

    # Check the exit code of the service.
    local exit_code=$(docker-compose -f "$script_dir"/docker-compose.yaml ps -q "$service_name" | xargs docker inspect -f '{{.State.ExitCode}}')
    if [ $exit_code -ne 0 ]; then
      echo "$service_name failed with exit code $exit_code!"
      exit 1
    fi
  done
}

run_services
