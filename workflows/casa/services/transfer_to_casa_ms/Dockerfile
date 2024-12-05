FROM 1_makems_base:latest

# Prepare run
WORKDIR /dsa/code
COPY workflows/external/services/transfer_to_casa_ms/src src

# Directies that will be mounted
RUN mkdir -p /dsa/data
RUN mkdir -p /dsa/run

WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]