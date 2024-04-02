FROM base:latest

# Prepare run
WORKDIR /dsa/code
COPY services/simulate_ionosphere/src src

# Directies that will be mounted
RUN mkdir -p /dsa/run
WORKDIR /dsa/run


ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]