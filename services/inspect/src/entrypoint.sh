#!/bin/bash

# Run from /dsa/input

set -euo pipefail
set -x

# Start up a jupyter notebook which will be useful even while it's running

jupyter notebook --notebook-dir=notebooks --port=8888 --no-browser --allow-root \
  --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --NotebookApp.notebook_dir=notebooks \
  --NotebookApp.token="$JUPYTER_TOKEN" --NotebookApp.allow_password_change="False" \
  --NotebookApp.password=""
