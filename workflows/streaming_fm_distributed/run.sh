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

if [[ -z "$NODE_NAME" ]]; then
  echo "NODE_NAME is not set. Using the hostname as the node name."
  NODE_NAME=$(hostname)
fi

if [[ -z "$DSA_CONTENT_SSH_USERNAME" ]]; then
  echo "DSA_CONTENT_SSH_USERNAME is not set. Using the current user as the SSH username."
  DSA_CONTENT_SSH_USERNAME=$(whoami)
fi

# Use a subshell so that the exported variables are not available after the script finishes
(

  UID=$(id -u)
  GID=$(id -g)
  export UID
  export GID

  export NODE_NAME
  export DSA_CONTENT_SSH_USERNAME

  # Function to look up an IP from node_map based on a node name.
  lookup_ip() {
    local node_name="$1"
    if [[ ! -f node_map ]]; then
      echo "Error: node_map file does not exist." >&2
      exit 1
    fi

    local ip
    ip=$(awk -v name="$node_name" '$1 == name {print $2}' node_map)
    if [[ -z "$ip" ]]; then
      echo "Error: Node name '$node_name' not found in node_map." >&2
      exit 1
    fi
    echo "$ip"
  }

  # Set the Ray head IP, preferring the provided RAY_HEAD_IP over lookup.
  if [[ -z "$RAY_HEAD_IP" ]]; then
    if [[ -n "$RAY_HEAD_NODE_NAME" ]]; then
      RAY_HEAD_IP=$(lookup_ip "$RAY_HEAD_NODE_NAME")
    else
      RAY_HEAD_IP=
      echo "Error: Neither RAY_HEAD_IP nor RAY_HEAD_NODE_NAME is set."
    fi
  fi
  export RAY_HEAD_IP

  # Set the node IP address, preferring the provided NODE_IP_ADDRESS over lookup.
  if [[ -z "$NODE_IP_ADDRESS" ]]; then
    if [[ -n "$NODE_NAME" ]]; then
      NODE_IP_ADDRESS=$(lookup_ip "$NODE_NAME")
    else
      echo "NODE_IP_ADDRESS is not set. Trying to automatically find it."
      # Try getting the IP address from ifconfig.me, otherwise use the hostname's first listed IP address (which might be the internal IP)
      NODE_IP_ADDRESS=$(curl -4 ifconfig.me || hostname -I | awk '{print $1}' || exit 1)
    fi
  fi
  export NODE_IP_ADDRESS

  echo "Ray head IP: $RAY_HEAD_IP"
  echo "Node IP: $NODE_IP_ADDRESS"

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
