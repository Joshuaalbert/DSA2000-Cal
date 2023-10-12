#!/bin/bash

# gets the script dir, should work with bash <script> and source <script>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# We force rebuild to take into account edits to changes to .env variables between runs.
#DOCKER_BUILDKIT=0  --no-cache
docker compose -f "$SCRIPT_DIR"/docker-compose.yaml build prepare_run
docker compose -f "$SCRIPT_DIR"/docker-compose.yaml up -d --force-recreate prepare_run

# This streams the logs. Push ctrl-C to detach from logs.
docker compose logs -f
