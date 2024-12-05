# FROM python:3.11.7
FROM nvidia/cuda:12.3.0-base-ubuntu20.04

# Install Python 3.11 and required dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential gcc g++ \
    && apt-get clean

# Set Python 3.11 as the default Python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1


# Install py-spy and other packages for rest of package

RUN apt-get update && apt-get install -y gdb lldb sudo rsync

RUN pip install py-spy && chown root:root `which py-spy` && chmod u+s `which py-spy`

# Install the ducc0 package from source for performance
RUN pip install --no-binary ducc0 'ducc0==0.35.0' 'numpy<2'

# Install JAX for GPU
RUN pip install jax[cuda12_cudnn91] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# requirements rarely change, so install those here
COPY ./dsa2000_cal/requirements.txt /dsa/code/requirements.txt
COPY ./dsa2000_cal/requirements-notebooks.txt /dsa/code/requirements-notebooks.txt
RUN pip install -r /dsa/code/requirements.txt
RUN pip install -r /dsa/code/requirements-notebooks.txt


# Download the ephem data for astropy
RUN python -c 'import astropy.time as at; at.Time.now().ut1'

# Copy entrypoint
COPY ./workflows/streaming_fm_distributed/ray/src /dsa/code/src

# Mask ssh dir
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Make a writable directory for ssh
RUN mkdir -p /tmp/ssh && chmod 700 /tmp/ssh


# For data persistence in the run directory
RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]