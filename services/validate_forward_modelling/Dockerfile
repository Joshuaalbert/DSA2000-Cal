FROM 1_wsclean_base:latest


# Prepare run
WORKDIR /dsa/code
COPY services/validate_forward_modelling/src src

RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]