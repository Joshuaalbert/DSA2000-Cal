FROM python:3.10.8


RUN apt-get update && \
    apt-get install -y vim less git htop feh build-essential wget && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /dsa/code

# Prepare run
COPY services/predict_fft/src src
RUN ls && ls src && pip install -r src/requirements.txt

RUN mkdir -p /dsa/run
WORKDIR /dsa/run

