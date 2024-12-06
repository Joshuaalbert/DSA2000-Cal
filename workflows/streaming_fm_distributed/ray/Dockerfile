# FROM python:3.11.7
FROM nvidia/cuda:12.1.1-base-ubuntu20.04

# Install Python 3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam


# Install Python 3.11 and dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        tk-dev \
        uuid-dev \
        g++ \
        wget \
        && add-apt-repository ppa:deadsnakes/ppa -y; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv python3-pip; \
    apt-get remove -y python3.8 python3.8-minimal; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Reset DEBIAN_FRONTEND to prevent unintended behavior in the container
ENV DEBIAN_FRONTEND=dialog

# Install py-spy and other packages for rest of package

RUN apt-get update && apt-get install -y gdb lldb sudo rsync build-essential \
    g++

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