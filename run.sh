#!/bin/bash

# gets the script dir, should work with bash <script> and source <script>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

docker compose -f "$SCRIPT_DIR"/docker-compose.yaml down

docker compose -f "$SCRIPT_DIR"/docker-compose.yaml build --no-cache quartical
docker compose -f "$SCRIPT_DIR"/docker-compose.yaml up -d --force-recreate quartical

# This streams the logs. Push ctrl-C to detach from logs.
docker compose logs -f
