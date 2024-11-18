#!/bin/bash

# Fail on first error
set -e

# Usage: ./run.sh KEY1=VALUE1 KEY2=VALUE2 ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

# Check if .env file exists
COMMON_ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$COMMON_ENV_FILE" ]]; then
  echo "Error: .env file not found at $COMMON_ENV_FILE"
  exit 1
fi

for var in "$@"; do
  export "${var%=*}"="${var#*=}"
done

export IS_RAY_HEAD=false

# Use the temporary .env file in Docker Compose commands
echo "Tearing down old services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" down

echo "Building the services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" build

echo "Configuring the services..."
docker compose config

echo "Starting the services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" up -d ray

docker compose logs -f

## Clean up the temporary .env file
#rm -f "$TEMP_ENV_FILE"
