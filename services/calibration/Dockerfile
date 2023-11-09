FROM base:latest

# Prepare run
WORKDIR /dsa/code
COPY services/calibration/src src
RUN pip install --upgrade pip && pip install -r src/requirements.txt


RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]

