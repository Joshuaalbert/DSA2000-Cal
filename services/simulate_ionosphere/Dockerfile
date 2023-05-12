FROM python:3.10.8

# install extras

RUN apt-get update && \
    apt-get install -y vim less ffmpeg git htop feh mplayer build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# container directories that will be mounted
RUN mkdir -p /dsa/run

# Install bayes_gain_screens

WORKDIR /dsa/code

RUN git clone https://github.com/Joshuaalbert/tomographic_kernel.git \
    && pip install -r tomographic_kernel/requirements.txt \
    && pip install tomographic_kernel/.

RUN pip freeze

COPY services/simulate_ionosphere/src/entrypoint.sh .
COPY services/simulate_ionosphere/src/sky_model.bbs .

# We run from within input folder

WORKDIR /dsa/run


ENTRYPOINT [ "/bin/bash", "/dsa/code/entrypoint.sh" ]