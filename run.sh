#!/bin/bash

set -euo pipefail
set -x

declare -A cache_options
declare -A volumes
declare -A gpu_services
declare -A extras

services_order=(
  #    "inspect"
  #  "prepare_run"
  #  "simulate_ionosphere"
  #  "compute_beam"
  #  "simulate_instrumental_effects"
  "predict_dft"
  #"predict_fft"
  #  "simulate_rfi"
  "sum_visibilities"
  #  "dirty_image"
  "image_uncorrected"
  #  "image_corrected_perfect"
  #  "dirty_image"
  #  "calibration"
  #  "image_a_proj"
)

# The values represent whether to use --no-cache or not. 1 for yes, 0 for no.
cache_options=(
  ["prepare_run"]=0
  ["simulate_ionosphere"]=0
  ["compute_beam"]=0
  ["simulate_instrumental_effects"]=0
  ["predict_dft"]=0
  ["predict_fft"]=0
  ["simulate_rfi"]=0
  ["sum_visibilities"]=0
  ["dirty_image"]=0
  ["image_uncorrected"]=0
  ["calibration"]=0
  ["image_a_proj"]=0
)

# Add special volumes that need to be mounted
# Use single quotes for variables.
volumes=(
  ["prepare_run"]='-v $DATA_DIR_HOST:/dsa/data'
)

# Add port forwarding maps, or environment variables (use single quotes to do delayed expand).
extras=(
  ["inspect"]='-p 8890:8888 -e JUPYTER_TOKEN=$JUPYTER_TOKEN'
)

# Add other services requiring GPU with a value of 1.
gpu_services=(
  ["predict_fft"]=0
  ["dirty_image"]=0
  ["image_uncorrected"]=0
)

expand_string_from_env() {
  local env_file="$1"
  local input_string="$2"

  # Use a subshell to keep sourcing of .env localized
  (
    # Source the .env file within the subshell
    if [ -f "$env_file" ]; then
      source "$env_file"
    fi
    # Evaluate the string with environment variables expanded
    eval echo "$input_string"
  )
}

# Function to retrieve the value from an associative array with a default fallback, and then expand any environment variables in the value
get_expanded_value_or_default() {
  local -n assoc_array="$1" # Use a nameref for the associative array
  local key="$2"
  local default_value="$3"
  local env_file="$4"

  if [[ ${assoc_array[$key]+isset} ]]; then
    echo $(expand_string_from_env "$env_file" "${assoc_array[$key]}")
  else
    echo "$default_value"
  fi
}

run_services() {
  local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

  # Source .env file
  env_file="$script_dir/.env"

  if [ -f "$env_file" ]; then
    source "$env_file"
  else
    echo ".env file not found!"
    exit 1
  fi

  for service_name in "${services_order[@]}"; do
    echo "==== Processing service: $service_name ===="

    local cache_option=""
    if [ ${cache_options["$service_name"]+isset} ]; then
      if [ ${cache_options["$service_name"]} -eq 1 ]; then
        cache_option="--no-cache"
      fi
    fi

    # Build the Docker image
    echo "Building Docker image for $service_name..."
    docker build $cache_option -t "$service_name" -f "$script_dir/services/$service_name/Dockerfile" "$script_dir" || {
      echo "Failed to build Docker image for $service_name"
      exit 1
    }

    # Check if the service needs GPU
    local gpu_option=""
    if [ ${gpu_services["$service_name"]+isset} ]; then
      if [ ${gpu_services["$service_name"]} -eq 1 ]; then
        gpu_option="--gpus all"
      fi
    fi

    # Construct the volume arguments
    local volume_args=("-v" "/dev/shm:/dev/shm" "-v" "$RUN_DIR_HOST:/dsa/run")
    if [[ ${volumes["$service_name"]+isset} ]]; then
      if [[ -n ${volumes["$service_name"]} ]]; then
        volume_args+=(
          $(get_expanded_value_or_default volumes "$service_name" "" "$env_file")
        )
      fi
    fi

    # Construct the extra arguments
    local extra_args=()
    if [[ ${extras["$service_name"]+isset} ]]; then
      if [[ -n ${extras["$service_name"]} ]]; then
        extra_args+=(
          $(get_expanded_value_or_default extras "$service_name" "" "$env_file")
        )
      fi
    fi

    # Remove the container if it already exists
    docker rm -f "$service_name" || true

    # Run the container as a daemon
    echo "Starting container for $service_name..."
    local container_id=$(docker run -d $gpu_option --name "$service_name" --env-file "$env_file" "${volume_args[@]}" "${extra_args[@]}" "$service_name")

    # Stream the logs until the service ends.
    docker logs -f "$container_id"

    # Check the exit code of the container.
    local exit_code=$(docker inspect "$container_id" --format='{{.State.ExitCode}}')
    if [ $exit_code -ne 0 ]; then
      echo "$service_name failed with exit code $exit_code!"
      exit 1
    fi

    # Clean up the container
    docker rm "$container_id"
  done
}

# Build or rebuild the base images as needed
build_base_images() {
  local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
  local base_dockerfiles_path="$script_dir/bases"
  local base_image_name=""
  local base_dockerfile=""

  # Loop over all the Dockerfiles in the base_images directory
  for base_dockerfile in "$base_dockerfiles_path"/*.Dockerfile; do
    # Extract the base image name from the Dockerfile filename
    base_image_name="$(basename "$base_dockerfile" .Dockerfile)"

    # Build the base image
    echo "Building or updating base image $base_image_name..."
    docker build -t "$base_image_name" -f "$base_dockerfile" "$script_dir" || {
      echo "Failed to build base image $base_image_name"
      exit 1
    }
  done
}

build_base_images

run_services
