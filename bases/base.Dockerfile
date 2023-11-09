FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim less ffmpeg git htop feh build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# install dsa2000_cal
WORKDIR /dsa/code

COPY dsa2000_cal dsa2000_cal
RUN pip install -r dsa2000_cal/requirements.txt && \
    pip install ./dsa2000_cal