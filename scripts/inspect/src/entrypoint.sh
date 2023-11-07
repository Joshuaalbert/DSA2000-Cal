#!/bin/bash

# Run server from /dsa/jupyter_root

set -euo pipefail
set -x

# Start up a jupyter notebook which will be useful even while it's running

jupyter notebook \
  --port=8888 --no-browser --allow-root \
  --ServerApp.allow_origin='*' \
  --ServerApp.ip='0.0.0.0' \
  --ServerApp.root_dir='/dsa/jupyter_root' \
  --IdentityProvider.token="$JUPYTER_TOKEN" \
  --PasswordIdentityProvider.allow_password_change="False" \
  --ServerApp.password=""
