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

# Create a temporary .env file
TEMP_ENV_FILE="$SCRIPT_DIR/.full_env"

# empty the file
if [ -f "$TEMP_ENV_FILE" ]; then
  rm "$TEMP_ENV_FILE"
fi

#cp "$COMMON_ENV_FILE" "$TEMP_ENV_FILE"

# Array to track variable names
declare -a ENV_VARS=()

# Load common environment variables
while IFS= read -r LINE; do
  # Extract key and value
  KEY="${LINE%%=*}"
  VALUE="${LINE#*=}"
  echo "Processing $KEY = $VALUE"
  # Only append if value is not empty
  if [[ -z "$VALUE" ]]; then
    continue
  fi
  echo "$KEY=$VALUE" >>"$TEMP_ENV_FILE"
  echo "Appended to ${TEMP_ENV_FILE}: $KEY=$VALUE"
  ENV_VARS+=("$KEY")
done <"$COMMON_ENV_FILE"

# Append dynamic arguments to the ENV_VARS
for ARG in "$@"; do
  # Extract key and value
  KEY="${ARG%%=*}"
  VALUE="${ARG#*=}"
  echo "$KEY=$VALUE" >>"$TEMP_ENV_FILE"
  echo "Appended to ${TEMP_ENV_FILE}: $KEY=$VALUE"
  ENV_VARS+=("$KEY")
done

cat "$TEMP_ENV_FILE"

# Use the temporary .env file in Docker Compose commands
echo "Tearing down old services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" down --env-file "$TEMP_ENV_FILE"

echo "Building the services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" build --env-file "$TEMP_ENV_FILE"

echo "Configuring the services..."
docker compose config

echo "Starting the services..."
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" up -d --env-file "$TEMP_ENV_FILE"

docker compose logs -f

## Clean up the temporary .env file
#rm -f "$TEMP_ENV_FILE"
