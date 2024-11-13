FROM 1_base:latest


# Prepare run
WORKDIR /dsa/run
COPY workflows/streaming_fm/src src

# Directies that will be mounted
RUN mkdir -p /dsa/run

WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]