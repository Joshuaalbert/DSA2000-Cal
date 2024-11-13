#!/bin/bash

# Default values in case the arguments are not provided
NUM_PROCESSES=1
PROCESS_ID=0
COORDINATOR_ADDRESS="127.0.0.1"
PLOT_FOLDER="plots"
GIT_BRANCH="main"  # Default branch

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --head)
        IS_HEAD=true
        shift # past argument
        ;;
        --head_ip)
        COORDINATOR_ADDRESS="$2"
        shift # past argument
        shift # past value
        ;;
        --num_processes)
        NUM_PROCESSES="$2"
        shift # past argument
        shift # past value
        ;;
        --process_id)
        PROCESS_ID="$2"
        shift # past argument
        shift # past value
        ;;
        --plot_folder)
        PLOT_FOLDER="$2"
        shift # past argument
        shift # past value
        ;;
        --git_branch)
        GIT_BRANCH="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        # Unknown option
        shift # past argument
        ;;
    esac
done

# Clone or pull the repository
REPO_DIR="DSA2000-Cal"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repository already exists. Pulling latest changes from branch $GIT_BRANCH..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$GIT_BRANCH"
    git pull origin "$GIT_BRANCH"
else
    echo "Cloning repository from branch $GIT_BRANCH..."
    git clone --branch "$GIT_BRANCH" https://github.com/Joshuaalbert/DSA2000-Cal.git "$REPO_DIR"
fi

# Install the code
echo "Installing the DSA2000-Cal package..."
pip install "$REPO_DIR"/dsa2000_cal

# Start Ray
if [ "$IS_HEAD" = true ]; then
    echo "Starting Ray head node..."
    RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0 ray start --head \
      --port=6379 --redis-shard-ports=6380,6381 --object-manager-port=22345 --node-manager-port=22346 \
      --dashboard-host=0.0.0.0 --metrics-export-port=8090
    COORDINATOR_ADDRESS="127.0.0.1"
    PROCESS_ID=0
else
    if [ -z "$COORDINATOR_ADDRESS" ]; then
        echo "Error: --head_ip must be specified for worker nodes."
        exit 1
    fi
    echo "Starting Ray worker node connecting to head at $COORDINATOR_ADDRESS..."
    RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0 ray start --address="${COORDINATOR_ADDRESS}:6379" \
      --object-manager-port=22345 --node-manager-port=22346
    if [ -z "$PROCESS_ID" ]; then
        echo "Error: --process_id must be specified for worker nodes."
        exit 1
    fi
fi

# Create plot folder if it doesn't exist
mkdir -p "$PLOT_FOLDER"

# Launch the Python process
echo "Launching process with:"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "PROCESS_ID=${PROCESS_ID}"
echo "COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS}"
echo "PLOT_FOLDER=${PLOT_FOLDER}"
echo "GIT_BRANCH=${GIT_BRANCH}"

JAX_PLATFORMS=cpu python /dsa/code/DSA2000-Cal/scripts/streaming_forward_modelling/launch_process.py \
  --num_processes="${NUM_PROCESSES}" \
  --process_id="${PROCESS_ID}" \
  --coordinator_address="${COORDINATOR_ADDRESS}" \
  --plot_folder="${PLOT_FOLDER}"
