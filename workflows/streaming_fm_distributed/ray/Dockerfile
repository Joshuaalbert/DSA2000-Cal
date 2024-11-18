FROM python:3.11.7

RUN apt-get update && apt-get install -y gdb lldb sudo
RUN pip install py-spy && chown root:root `which py-spy` && chmod u+s `which py-spy`

# requirements rarely change, so install those here
COPY ./DSA2000-Cal/dsa2000_cal/requirements.txt /dsa/code/requirements.txt
COPY ./DSA2000-Cal/dsa2000_cal/requirements-notebooks.txt /dsa/code/requirements-notebooks.txt
RUN pip install -r /dsa/code/requirements.txt
RUN pip install -r /dsa/code/requirements-notebooks.txt

# Copy entrypoint
COPY ./workflows/streaming_fm_distributed/ray/src /dsa/code/src

# For data persistence in the run directory
RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]