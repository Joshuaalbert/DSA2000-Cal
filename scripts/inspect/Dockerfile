FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim htop less git && \
    rm -rf /var/lib/apt/lists/*


COPY services/inspect/src src
RUN pip install -r src/requirements.txt

# Directies that will be mounted
RUN mkdir -p /dsa/run


# Setup Jupyter root
RUN mkdir /dsa/jupyter_root \
    && ln -s /dsa/code/src/notebooks/ /dsa/jupyter_root/notebooks \
    && ln -s /dsa/run/ /dsa/jupyter_root/run

WORKDIR /dsa/jupyter_root

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]