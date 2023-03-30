FROM python:3.10.8

WORKDIR /code

# container directories that will be mounted
RUN mkdir /gains && mkdir /logs && mkdir /data

# Install quartical and LSM stuff

ADD services/quartical/requirements.txt /code

RUN pip install -r requirements.txt

COPY services/quartical/src/entrypoint.sh .
COPY $PARSET .
COPY $SKY_MODEL_TIGGER .

RUN pip freeze

ENTRYPOINT [ "/bin/bash", "entrypoint.sh" ]