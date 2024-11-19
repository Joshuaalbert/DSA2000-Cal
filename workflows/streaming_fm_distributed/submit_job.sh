#!/bin/bash

SERVICE_NAME="ray"

#docker compose exec $SERVICE_NAME /dsa/code/src/submit_job.sh

docker exec -d $SERVICE_NAME /dsa/code/src/submit_job.sh

docker logs -f $SERVICE_NAME
