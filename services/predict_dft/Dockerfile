FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim less ffmpeg git htop feh build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# container directories that will be mounted


# install dsa2000_cal
WORKDIR /dsa/code
COPY dsa2000_cal dsa2000_cal
RUN pip install -r dsa2000_cal/requirements.txt && \
    pip install ./dsa2000_cal


COPY services/predict_dft/src src

# Directies that will be mounted
RUN mkdir -p /dsa/run
WORKDIR /dsa/run


ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]