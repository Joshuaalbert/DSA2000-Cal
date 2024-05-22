FROM 1_base:latest

# Prepare run
WORKDIR /dsa/code
COPY workflows/forward_modelling/services/forward_model/src src

# Directies that will be mounted
RUN mkdir -p /dsa/data
RUN mkdir -p /dsa/run

WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]