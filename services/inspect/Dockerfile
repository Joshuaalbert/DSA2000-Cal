FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# container directories that will be mounted
RUN mkdir -p /dsa/notebooks/output && mkdir /dsa/notebooks/data

# Install quartical and LSM stuff

WORKDIR /dsa/code

ADD services/inspect/requirements.txt .

RUN pip install --verbose -r requirements.txt


WORKDIR /dsa/code
COPY dsa2000_cal .
RUN pip install -r dsa2000_cal/requirments.txt && \
    pip install dsa2000_cal

RUN pip freeze

COPY services/inspect/src/entrypoint.sh .

# We run from within input folder

WORKDIR /dsa/notebooks
COPY services/inspect/src/notebooks/ .


ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]