#!/bin/bash

#!/bin/bash

# Clone or pull the repository
REPO_DIR="DSA2000-Cal"
PACKAGE_DIR="${REPO_DIR}/dsa2000_cal"
PLOT_FOLDER="plots"

if [ -d "$REPO_DIR/.git" ]; then
  echo "Repository already exists. Pulling latest changes from branch $GIT_BRANCH..."
  cd "$REPO_DIR" || exit 1
  git fetch origin
  git checkout "$GIT_BRANCH" || exit 1
  git pull origin "$GIT_BRANCH"
else
  echo "Cloning repository from branch $GIT_BRANCH..."
  git clone --branch "$GIT_BRANCH" https://github.com/Joshuaalbert/DSA2000-Cal.git "$REPO_DIR"
fi

# Install the code
echo "Installing the $REPO_DIR package..."
pip install -e "$PACKAGE_DIR"

# Create plot folder if it doesn't exist
mkdir -p "$PLOT_FOLDER"

# Launch the Python process
echo "Launching process with:"
echo "IS_RAY_HEAD=${IS_RAY_HEAD}"
echo "RAY_HEAD_IP=${RAY_HEAD_IP}"
echo "RAY_REDIS_PORT=${RAY_REDIS_PORT}"
echo "GIT_BRANCH=${GIT_BRANCH}"
echo "PACKAGE_DIR=${PACKAGE_DIR}"

# Start Ray
if [ "$IS_RAY_HEAD" = true ]; then
  echo "Starting Ray head node..."
  RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0 ray start --head \
    --port=6379 --redis-shard-ports=6380,6381 --object-manager-port=22345 --node-manager-port=22346 \
    --dashboard-host=0.0.0.0 --metrics-export-port=8090

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
  RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0 ray start --address="${RAY_HEAD_IP}:${RAY_REDIS_PORT}" \
    --object-manager-port=22345 --node-manager-port=22346
fi

tail -f /dev/null
