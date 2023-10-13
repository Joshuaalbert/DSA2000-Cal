FROM python:3.10.8


RUN apt-get update && \
    apt-get install -y vim less git htop feh build-essential wget && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /dsa/code

COPY requirements.txt .

RUN pip install -r requirements.txt
