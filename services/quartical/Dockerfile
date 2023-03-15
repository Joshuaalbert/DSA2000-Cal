FROM python:3.10.8

WORKDIR /code

# container directories that will be mounted
RUN mkdir /gains && mkdir /logs && /data

# Install quartical and LSM stuff

ADD requirements.txt /code

RUN pip install -r requirements.txt

COPY src/entrypoint.sh .
COPY $PARSET .

ENTRYPOINT [ "/bin/bash", "entrypoint.sh" ]