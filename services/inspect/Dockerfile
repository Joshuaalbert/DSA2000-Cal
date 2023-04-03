FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# container directories that will be mounted
RUN mkdir -p /dsa/notebooks/output && mkdir /dsa/notebooks/data

# Install quartical and LSM stuff

WORKDIR /dsa/code

ADD services/quartical/requirements.txt .

RUN pip install --verbose -r requirements.txt

RUN pip freeze

COPY services/quartical/src/entrypoint.sh .

# We run from within input folder

WORKDIR /dsa/notebooks
COPY services/quartical/src/notebooks/ .


ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]