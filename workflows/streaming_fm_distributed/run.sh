#!/bin/bash

# Fail on first error
set -e

# Usage: ./script.sh KEY1=VALUE1 KEY2=VALUE2 ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

# Check if .env file exists
COMMON_ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$COMMON_ENV_FILE" ]; then
  echo "Error: .env file not found at $COMMON_ENV_FILE"
  exit 1
fi

# Create a temporary .env file
TEMP_ENV_FILE="$SCRIPT_DIR/.env.temp"

# Start with common .env content
cp "$COMMON_ENV_FILE" "$TEMP_ENV_FILE"

# Array to track variable names
declare -a ENV_VARS

# Collect variable names from the common .env file
while IFS='=' read -r KEY _; do
  if [[ -n "$KEY" && ! "$KEY" =~ ^# ]]; then
    ENV_VARS+=("$KEY")
  fi
done <"$COMMON_ENV_FILE"

# Append dynamic arguments to the .env file
for ARG in "$@"; do
  echo "$ARG" >>"$TEMP_ENV_FILE"
  # Add the variable name to the array
  KEY=$(echo "$ARG" | cut -d= -f1)
  ENV_VARS+=("$KEY")
done

# Ensure all environment variables are non-empty
for VAR in "${ENV_VARS[@]}"; do
  # Use `grep` to find the last occurrence of the variable in the .env file
  VALUE=$(grep -oE "^$VAR=.*" "$TEMP_ENV_FILE" | tail -n1 | cut -d= -f2-)
  if [[ -z "$VALUE" ]]; then
    echo "Error: Environment variable '$VAR' is not set or empty in $TEMP_ENV_FILE"
    rm -f "$TEMP_ENV_FILE"
    exit 1
  fi
done

# Set up local Docker cache directory
CACHE_DIR="$SCRIPT_DIR/docker_cache"
mkdir -p "$CACHE_DIR"

export DOCKER_BUILDKIT=1


# Use the temporary .env file in Docker Compose commands
docker-compose --env-file "$TEMP_ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yaml" down
docker-compose --env-file "$TEMP_ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yaml" build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from=type=local,src="$CACHE_DIR" \
  --cache-to=type=local,dest="$CACHE_DIR"
docker-compose --env-file "$TEMP_ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yaml" up -d

# Clean up the temporary .env file
rm -f "$TEMP_ENV_FILE"

docker-compose logs -f
