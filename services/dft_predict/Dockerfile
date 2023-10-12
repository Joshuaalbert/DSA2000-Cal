FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# container directories that will be mounted
RUN mkdir -p /dsa/run && mkdir /dsa/data

# Install quartical and LSM stuff

WORKDIR /dsa/code

ADD services/dft_predict/requirements.txt .

RUN pip install --verbose -r requirements.txt

RUN pip freeze

COPY services/dft_predict/src/ .


ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]