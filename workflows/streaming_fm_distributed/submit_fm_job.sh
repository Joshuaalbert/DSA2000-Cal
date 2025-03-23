#!/bin/bash

SERVICE_NAME="ray_head"

docker compose exec $SERVICE_NAME /dsa/code/src/submit_fm_job.sh
