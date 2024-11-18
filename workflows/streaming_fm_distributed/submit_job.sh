#!/bin/bash

SERVICE_NAME="ray"

docker compose exec $SERVICE_NAME /dsa/code/src/submit_job.sh
