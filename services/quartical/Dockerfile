FROM python:3.10.8

WORKDIR /code

# container directories that will be mounted
RUN mkdir /gains && mkdir /logs

# Install quartical and LSM stuff

ADD requirements.txt /code

RUN pip install -r requirements.txt

COPY .. /code
COPY $RUN_DIR/parset.yaml /code/parset.yaml

ENTRYPOINT [ "/bin/bash", "entrypoint.sh" ]