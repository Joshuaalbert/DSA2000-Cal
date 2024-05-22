FROM 1_base:latest

# Prepare run
WORKDIR /dsa/code
COPY workflows/calibration/services/lwa_calibration/src src

# Directies that will be mounted
RUN mkdir -p /dsa/data
RUN mkdir -p /dsa/run

WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]