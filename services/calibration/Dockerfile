FROM python:3.10.8

RUN apt-get update && \
    apt-get install -y vim less git htop feh wget build-essential gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*



WORKDIR /dsa/code

# Prepare run
COPY services/calibration/src src
RUN pip install --upgrade pip && pip install -r src/requirements.txt

# install dsa2000_cal which is on host at ../../dsa2000_cal
COPY dsa2000_cal dsa2000_cal
RUN pip install -r dsa2000_cal/requirements.txt && \
    pip install ./dsa2000_cal

RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]

