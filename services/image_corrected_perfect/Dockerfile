FROM wsclean_base:latest

# Prepare run
WORKDIR /dsa/code
COPY services/image_corrected_perfect/src src

RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]