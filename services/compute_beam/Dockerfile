FROM base:latest

# install dsa2000_cal
WORKDIR /dsa/code
COPY services/compute_beam/src src

# Directies that will be mounted
RUN mkdir -p /dsa/run
WORKDIR /dsa/run


ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]