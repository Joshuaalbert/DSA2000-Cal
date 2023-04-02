FROM python:3.10.8

# container directories that will be mounted
RUN mkdir -p /dsa/run && mkdir /dsa/data && mkdir /dsa/input

# Install quartical and LSM stuff

WORKDIR /dsa/code

ADD services/quartical/requirements.txt .

RUN pip install -r requirements.txt

RUN pip freeze

COPY services/quartical/src/entrypoint.sh .

# We run from within input folder

WORKDIR /dsa/input


ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]