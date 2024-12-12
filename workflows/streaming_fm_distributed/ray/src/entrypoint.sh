#!/bin/bash


# Launch the Python process
echo "Launching process with:"
echo "IS_RAY_HEAD=${IS_RAY_HEAD}"
echo "RAY_HEAD_IP=${RAY_HEAD_IP}"
echo "NODE_IP_ADDRESS=${NODE_IP_ADDRESS}"

TEMP_DIR="/dsa/run/temp"
mkdir -p "$TEMP_DIR"
PACKAGE_DIR="/dsa/code/package"

# Ensure it points to valid pyproject.toml
if [ ! -f "$PACKAGE_DIR/pyproject.toml" ]; then
  echo "Error: pyproject.toml not found at $PACKAGE_DIR"
  exit 1
fi

## Set up tmp .ssh directory
#mkdir -p /tmp/.ssh
#cp /root/.ssh/* /tmp/.ssh/
#chmod 700 /tmp/.ssh
#chmod 600 /tmp/.ssh/*
#git config --global core.sshCommand "ssh -F /tmp/.ssh/config"
# Use it, then delete
#rm -rf /tmp/.ssh

# Install the code
echo "Installing the $PACKAGE_DIR package..."
pip install -e "$PACKAGE_DIR"[crypto,stocks]


# Start Ray

# This is required to prevent heatbeat timeouts in the Ray cluster: https://github.com/ray-project/ray/issues/45179#issuecomment-2191816159
LARGE_N=99999999999
export RAY_health_check_initial_delay_ms=$LARGE_N
export RAY_health_check_period_ms=$LARGE_N
NUMEXPR_MAX_THREADS=$(nproc)
export NUMEXPR_MAX_THREADS

if [ -z "$IS_RAY_HEAD" ]; then
  echo "Error: IS_RAY_HEAD must be specified."
  exit 1
fi

if [ -z "$NODE_IP_ADDRESS" ]; then
  echo "Error: NODE_IP_ADDRESS must be specified."
  exit 1
fi
if [ -z "$NODE_NAME" ]; then
  echo "Error: NODE_NAME must be specified."
  exit 1
fi

echo "Node IP address: $NODE_IP_ADDRESS ($NODE_NAME)"

if [ "$IS_RAY_HEAD" = true ]; then
  echo "Starting Ray head node..."

  ray start --head \
    --port=6379 \
    --metrics-export-port=8090 \
    --dashboard-host=0.0.0.0 \
    --temp-dir=$TEMP_DIR \
    --ray-client-server-port=10001 \
    --redis-shard-ports=6380,6381 \
    --node-manager-port=12345 \
    --object-manager-port=12346 \
    --runtime-env-agent-port=12347 \
    --dashboard-agent-grpc-port=12348 \
    --dashboard-agent-listen-port=52365 \
    --dashboard-port=8265 \
    --dashboard-grpc-port=50052 \
    --min-worker-port=20000 \
    --max-worker-port=20100 \
    --node-ip-address=$NODE_IP_ADDRESS \
    --node-name=$NODE_NAME

  ray status

  # Start Jupyter Notebook only on the head node
  JUPYTER_ROOT_DIR="$PACKAGE_DIR/notebooks"
  jupyter notebook \
    --port=8888 --no-browser --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.ip='0.0.0.0' \
    --ServerApp.root_dir="${JUPYTER_ROOT_DIR}" \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --PasswordIdentityProvider.allow_password_change="False" \
    --ServerApp.password="" \
    &

  jupyter server list

  streamlit run "${PACKAGE_DIR}/dashboards/dsa/home.py" --server.port 8501 &

  service cron start

  # chmod +x /dsa/code/src/cleanup_logs.sh
  # (crontab -l 2>/dev/null; echo "0 * * * * /dsa/code/cleanup_logs.sh") | crontab -
  chmod +x /dsa/code/src/scrape_metric_targets.py
  # Run every minute
  (crontab -l 2>/dev/null; echo "* * * * * python /dsa/code/src/scrape_metric_targets.py") | crontab -
  #service cron start

else
  if [ -z "$RAY_HEAD_IP" ]; then
    echo "Error: RAY_HEAD_IP must be specified for worker nodes."
    exit 1
  fi
  echo "Starting Ray worker node connecting to head at ${RAY_HEAD_IP}..."

  ray start --address="${RAY_HEAD_IP}:6379" \
    --metrics-export-port=8090 \
    --node-manager-port=12345 \
    --object-manager-port=12346 \
    --runtime-env-agent-port=12347 \
    --dashboard-agent-grpc-port=12348 \
    --dashboard-agent-listen-port=52365 \
    --min-worker-port=20000 \
    --max-worker-port=20100 \
    --node-ip-address=$NODE_IP_ADDRESS \
    --node-name=$NODE_NAME

  ray status
fi

tail -f /dev/null
