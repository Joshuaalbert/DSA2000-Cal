#!/bin/bash

#!/bin/bash

# Clone or pull the repository
REPO_DIR="DSA2000-Cal"
PACKAGE_DIR="${REPO_DIR}/dsa2000_cal"
TEMP_DIR="/dsa/run/temp"
mkdir -p "$TEMP_DIR"

# make absolution

if [ -d "$REPO_DIR/.git" ]; then
  echo "Repository already exists. Pulling latest changes from branch $GIT_BRANCH..."
  cd "$REPO_DIR" || exit 1
  git fetch origin && git checkout "$GIT_BRANCH" && git pull origin "$GIT_BRANCH"
  cd - || exit 1
else
  echo "Cloning repository from branch $GIT_BRANCH..."
  git clone --branch "$GIT_BRANCH" https://github.com/Joshuaalbert/DSA2000-Cal.git "$REPO_DIR"
fi

# Install the code
echo "Installing the $REPO_DIR package..."
pip install -e "$PACKAGE_DIR"

# Launch the Python process
echo "Launching process with:"
echo "IS_RAY_HEAD=${IS_RAY_HEAD}"
echo "RAY_HEAD_IP=${RAY_HEAD_IP}"
echo "RAY_REDIS_PORT=${RAY_REDIS_PORT}"
echo "GIT_BRANCH=${GIT_BRANCH}"
echo "PACKAGE_DIR=${PACKAGE_DIR}"

# Start Ray

# This is required to prevent heatbeat timeouts in the Ray cluster: https://github.com/ray-project/ray/issues/45179#issuecomment-2191816159
LARGE_N=99999999999
export RAY_health_check_initial_delay_ms=$LARGE_N
export RAY_health_check_period_ms=$LARGE_N

if [ -z "$IS_RAY_HEAD" ]; then
  echo "Error: IS_RAY_HEAD must be specified."
  exit 1
fi

if [ "$IS_RAY_HEAD" = true ]; then
  echo "Starting Ray head node..."
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --metrics-export-port=8090 --temp-dir=$TEMP_DIR

  ray status

  # Start Jupyter Notebook only on the head node
  ROOT_DIR="/dsa/run/${REPO_DIR}/dsa2000_cal/notebooks"
  jupyter notebook \
    --port=8888 --no-browser --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.ip='0.0.0.0' \
    --ServerApp.root_dir="${ROOT_DIR}" \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --PasswordIdentityProvider.allow_password_change="False" \
    --ServerApp.password="" \
    &

  jupyter server list

  #chmod +x /run/code/cleanup_logs.sh
  #(crontab -l 2>/dev/null; echo "0 * * * * /run/code/cleanup_logs.sh") | crontab -
  #service cron start

else
  if [ -z "$RAY_HEAD_IP" ]; then
    echo "Error: RAY_HEAD_IP must be specified for worker nodes."
    exit 1
  fi
  if [ -z "$RAY_REDIS_PORT" ]; then
    echo "Error: RAY_REDIS_PORT must be specified for worker nodes."
    exit 1
  fi
  echo "Starting Ray worker node connecting to head at ${RAY_HEAD_IP}:${RAY_REDIS_PORT}..."
  ray start --address="$RAY_HEAD_IP:${RAY_REDIS_PORT}"
  ray status
fi

tail -f /dev/null
