#!/bin/bash

# Default values
NUM_PROCESSES=
PROCESS_ID=
COORDINATOR_ADDRESS=
PLOT_FOLDER=

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
        *)
        # Unknown option
        shift # past argument
        ;;
    esac
done

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
PLOT_FOLDER="/dsa/run/${PLOT_FOLDER}"
mkdir -p "$PLOT_FOLDER"

# Launch the Python process
echo "Launching process with:"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "PROCESS_ID=${PROCESS_ID}"
echo "COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS}"
echo "PLOT_FOLDER=${PLOT_FOLDER}"

JAX_PLATFORMS=cpu python /dsa/code/dsa2000_cal/scripts/steaming_forward_modelling/launch_process.py \
  --num_processes="${NUM_PROCESSES}" \
  --process_id="${PROCESS_ID}" \
  --coordinator_address="${COORDINATOR_ADDRESS}" \
  --plot_folder="${PLOT_FOLDER}"

#tail -f /dev/null
