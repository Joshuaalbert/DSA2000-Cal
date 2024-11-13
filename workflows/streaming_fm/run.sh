#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

# Configuration
IMAGE_NAME="streaming_fm"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile" # Path to your Dockerfile
BRIDGE_NETWORK_NAME="ray_network"
CONTAINER_RUN_DIR="/dsa/run" # Directory inside the container where data is mounted
GIT_BRANCH="joshs-working-branch"

# List of nodes with their corresponding working directories
# Format: "IP WORK_DIR"
nodes=(
  "mario /fastpool/albert/streaming_fm"
)
  #"wario /fastpool/albert/streaming_fm"

NUM_PROCESSES=${#nodes[@]}

HEAD_NODE_NAME="mario" # The hostname of the head node

# Resolve the hostname to an IP address
HEAD_NODE_IP=$(getent hosts "$HEAD_NODE_NAME" | awk '{ print $1 }')

if [ -z "$HEAD_NODE_IP" ]; then
  echo "Error: Unable to resolve hostname $HEAD_NODE_NAME to an IP address."
  exit 1
fi

echo "Resolved head node hostname ($HEAD_NODE_NAME) to IP: $HEAD_NODE_IP"

# Function to start containers
start_containers() {
  # Enable error handling
  set -e

  # Step 1: Build Docker Image from the root of repository
  docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$SCRIPT_DIR/../.."

  # Step 2: Create Docker bridge network on head node if it doesn't exist
  if ! docker network ls | grep -q "$BRIDGE_NETWORK_NAME"; then
    docker network create -d bridge "$BRIDGE_NETWORK_NAME"
  else
    echo "Docker network '$BRIDGE_NETWORK_NAME' already exists on head node."
  fi

  # Step 3: Save Docker image to a tar file on the head node
  IMAGE_TAR_PATH="/tmp/$IMAGE_NAME.tar"
  docker save "$IMAGE_NAME" -o "$IMAGE_TAR_PATH"

  # Step 4: Start Ray head node container
  docker rm -f "ray_head_container_${HOSTNAME}" >/dev/null 2>&1 || true

  # Find the working directory for the head node
  for node_info in "${nodes[@]}"; do
    read -r ip work_dir <<<"$node_info"
    if [ "$ip" == "$HEAD_NODE_NAME" ]; then
      HEAD_WORK_DIR="$work_dir"
      break
    fi
  done

  # Run the Ray head node container
  docker run -d --rm --network "$BRIDGE_NETWORK_NAME" --name "ray_head_container_${HOSTNAME}" \
    -v "$HEAD_WORK_DIR":"$CONTAINER_RUN_DIR" \
    "$IMAGE_NAME" --head \
    --num_processes="$NUM_PROCESSES" \
    --plot_folder="plots"

  # Step 5: Distribute Docker Image and Start Worker Containers
  process_id=1  # Start from 1 since 0 is the head node

  for node_info in "${nodes[@]}"; do
    read -r ip work_dir <<<"$node_info"

    if [ "$ip" != "$HEAD_NODE_NAME" ]; then
      {
        # Existing commands to set up worker nodes...

        # Run the Ray worker container passing the head node IP and process ID
        ssh "$ip" "docker run -d --rm --network '$BRIDGE_NETWORK_NAME' --name 'ray_worker_container_${ip}' \
          -v '$work_dir':'$CONTAINER_RUN_DIR' \
          '$IMAGE_NAME' --head_ip '$HEAD_NODE_IP' \
          --process_id '$process_id' \
          --num_processes '$NUM_PROCESSES' \
          --plot_folder 'plots' \
          --git_branch '$GIT_BRANCH'"
        process_id=$((process_id + 1))
      } || {
        echo "Error occurred while starting worker on $ip. Stopping all containers."
        stop_containers
        exit 1
      }
    fi
  done

  # Cleanup: Remove the image tar file from the head node
  rm "$IMAGE_TAR_PATH"
}


# Function to stop containers
stop_containers() {
  errors_occurred=false

  # Stop head node container
  if ! docker rm -f ray_head_container >/dev/null 2>&1; then
    echo "Failed to stop container on head node ($HEAD_NODE_NAME)"
    errors_occurred=true
  else
    echo "Stopped container on head node ($HEAD_NODE_NAME)"
  fi

  # Stop worker containers
  for node_info in "${nodes[@]}"; do
    read -r ip work_dir <<<"$node_info"

    if [ "$ip" != "$HEAD_NODE_NAME" ]; then
      if ssh "$ip" "docker rm -f ray_worker_container >/dev/null 2>&1"; then
        echo "Stopped container on worker node ($ip)"
      else
        echo "Failed to stop container on worker node ($ip)"
        errors_occurred=true
      fi
    fi
  done

  if [ "$errors_occurred" = true ]; then
    echo "Some containers failed to stop."
  else
    echo "All containers stopped successfully."
  fi
}

# Main script logic
if [ "$1" == "start" ]; then
  echo "Starting containers..."
  start_containers
elif [ "$1" == "stop" ]; then
  echo "Stopping containers..."
  stop_containers
else
  echo "Usage: $0 {start|stop}"
  exit 1
fi
