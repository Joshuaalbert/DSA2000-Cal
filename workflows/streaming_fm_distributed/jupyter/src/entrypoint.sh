#!/bin/bash

# Launch the Python process

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

NUMEXPR_MAX_THREADS=$(nproc)
export NUMEXPR_MAX_THREADS

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

tail -f /dev/null
