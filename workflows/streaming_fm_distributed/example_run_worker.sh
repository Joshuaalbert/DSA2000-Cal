#!/bin/bash

# Fail on first error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
echo "Script dir $SCRIPT_DIR"

WORKING_DIR=$PWD

$SCRIPT_DIR/run.sh \
  "IS_RAY_HEAD=false" \
  "RAY_HEAD_IP=mario" \
  "JUPYTER_TOKEN=1234" \
  "GF_SECURITY_ADMIN_PASSWORD=1234" \
  "RUN_DIR_HOST=$WORKING_DIR" \
  "GIT_BRANCH=main"

