FROM 1_wsclean_base:latest


# Prepare run
WORKDIR /dsa/code
COPY workflows/external/services/lwa_full_sky_image/src src

RUN mkdir -p /dsa/data
RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]