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

ENV_ARGS=""
for var in "$@"; do
  # append -e flag to each variable
  ENV_ARGS="$ENV_ARGS -e $var"
done

echo "Args passed: $ENV_ARGS"

# Use the temporary .env file in Docker Compose commands
echo "Tearing down old services..."
docker compose $ENV_ARGS -f "$SCRIPT_DIR/docker-compose.yaml" down

echo "Building the services..."
docker compose $ENV_ARGS -f "$SCRIPT_DIR/docker-compose.yaml" build

echo "Configuring the services..."
docker compose $ENV_ARGS config

echo "Starting the services..."
docker compose $ENV_ARGS -f "$SCRIPT_DIR/docker-compose.yaml" up -d

docker compose logs -f

## Clean up the temporary .env file
#rm -f "$TEMP_ENV_FILE"
