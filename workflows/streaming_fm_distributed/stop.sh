#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

NODE_NAME=$(hostname)

(
  export NODE_NAME

  # Try shutdown on head, else on worker
  docker compose exec ray_head /dsa/code/src/shutdown.sh || docker compose exec ray_worker /dsa/code/src/shutdown.sh

  sleep 5

  docker compose -f "$SCRIPT_DIR"/docker-compose.yaml down

  docker compose logs -f
)