FROM python:3.10.8

RUN apt-get update && \
    apt-get install -y vim less git htop feh wget build-essential gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*



WORKDIR /dsa/code

# Prepare run
COPY calibration/src src
RUN pip install --upgrade pip && pip install -r src/requirements.txt

# Directory that will be mounted to host to get data input
RUN mkdir -p /dsa/data

# Directory that will be mounted to host to run inside
RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]

