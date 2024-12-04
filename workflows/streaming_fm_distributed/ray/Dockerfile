FROM python:3.11.7

RUN apt-get update && apt-get install -y gdb lldb sudo rsync
RUN pip install py-spy && chown root:root `which py-spy` && chmod u+s `which py-spy`

# requirements rarely change, so install those here
COPY ./dsa2000_cal/requirements.txt /dsa/code/requirements.txt
COPY ./dsa2000_cal/requirements-notebooks.txt /dsa/code/requirements-notebooks.txt
RUN pip install -r /dsa/code/requirements.txt
RUN pip install -r /dsa/code/requirements-notebooks.txt

# Install the ducc0 package from source for performance
RUN pip install --ignore-installed --no-binary ducc0=0.35.0 --user ducc0

# Install JAX for GPU
RUN pip install --ignore-installed jax[jaxlib] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

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