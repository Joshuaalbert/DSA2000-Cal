FROM 1_base:latest

# Prepare run
WORKDIR /dsa/code
COPY workflows/external/services/inspect/src src
RUN pip install -r src/requirements.txt

# Directies that will be mounted
RUN mkdir -p /dsa/data
RUN mkdir -p /dsa/run

# Setup Jupyter root
RUN mkdir /dsa/jupyter_root \
    && ln -s /dsa/code/src/notebooks/ /dsa/jupyter_root/notebooks \
    && ln -s /dsa/run/ /dsa/jupyter_root/run

WORKDIR /dsa/jupyter_root

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]