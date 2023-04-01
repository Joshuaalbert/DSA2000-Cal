FROM python:3.10.8

# container directories that will be mounted
RUN mkdir /dsa/gains && mkdir /dsa/logs && mkdir /dsa/data && mkdir /dsa/input

# Install quartical and LSM stuff

WORKDIR /dsa/code

ADD services/quartical/requirements.txt .

RUN pip install -r requirements.txt

COPY services/quartical/src/entrypoint.sh .

# Copy input params

WORKDIR /dsa/input

RUN pip freeze

ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]