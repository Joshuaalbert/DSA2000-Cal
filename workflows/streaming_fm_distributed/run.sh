#!/bin/bash

# Fail on first error
set -e

# Usage: ./run.sh service1 service2 ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

# Check if .env file exists
COMMON_ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$COMMON_ENV_FILE" ]]; then
  echo "Error: .env file not found at $COMMON_ENV_FILE"
  exit 1
fi

# Ensure at least one service is passed
if [[ $# -lt 1 ]]; then
  echo "Error: No services specified. Usage: $0 SERVICE1 SERVICE2 ..."
  exit 1
fi

NODE_NAME=$(hostname)
DSA_CONTENT_SSH_USERNAME=$(whoami)

# Use a subshell so that the exported variables are not available after the script finishes
(

  # Only when on VPN trust this
  if [ -z "$NODE_IP_ADDRESS" ]; then
    echo "NODE_IP_ADDRESS is not set. Trying to automatically find it."
    # Try getting the IP address from ifconfig.me, otherwise use the hostname's first listed IP address (which might be the internal IP)
    NODE_IP_ADDRESS=$(curl -4 ifconfig.me || hostname -I | awk '{print $1}' || exit 1)
    export NODE_IP_ADDRESS
  fi
  # export so they are used in the Docker Compose file
  export NODE_NAME
  export DSA_CONTENT_SSH_USERNAME

  # Use the temporary .env file in Docker Compose commands
  echo "Tearing down old services..."
  docker compose -f "$SCRIPT_DIR/docker-compose.yaml" down

  echo "Building the services..."
  docker compose -f "$SCRIPT_DIR/docker-compose.yaml" build

  echo "Configuring the services..."
  docker compose config

  echo "Starting the services..."
  docker compose -f "$SCRIPT_DIR/docker-compose.yaml" up -d "$@"

  docker compose logs -f

  ## Clean up the temporary .env file
  #rm -f "$TEMP_ENV_FILE"
)
