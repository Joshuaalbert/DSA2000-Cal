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

if [ -d /root/.ssh_tmp ]; then
  # Create the destination if it doesn't exist
  mkdir -p /root/.ssh
  # Copy all contents from the temporary location
  cp -a /root/.ssh_tmp/. /root/.ssh/
  # Change ownership to root:root
  chown -R root:root /root/.ssh
fi

# Install the code
echo "Installing the $PACKAGE_DIR package..."
pip install -e "$PACKAGE_DIR"[geo,notebooks]

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


python /dsa/code/src/resource_logger.py &

python /dsa/code/src/scrape_metric_targets.py

tail -f /dev/null
