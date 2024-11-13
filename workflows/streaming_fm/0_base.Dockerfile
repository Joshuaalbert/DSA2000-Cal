FROM python:3.11

# install extras

RUN apt-get update && \
    apt-get install -y vim less ffmpeg git htop feh build-essential wget && \
    rm -rf /var/lib/apt/lists/*
