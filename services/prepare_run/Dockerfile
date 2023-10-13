# Using the same base image
FROM ubuntu:22.04

# Copy sources list and modify it



# Set some environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV GNUCOMPILER 10
ENV PYTHONVER 3.8
ENV DEB_SETUP_DEPENDENCIES \
    dpkg-dev \
    g++-$GNUCOMPILER \
    gcc-$GNUCOMPILER \
    libc-dev \
    cmake \
    gfortran-$GNUCOMPILER \
    git \
    wget

ENV DEB_DEPENCENDIES \
    python3-virtualenv \
    python3-pip \
    libfftw3-dev \
    python3-numpy \
    libfreetype6 \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-dev \
    libboost-all-dev \
    libcfitsio-dev \
    libhdf5-dev \
    wcslib-dev \
    libatlas-base-dev \
    liblapack-dev \
    python3-tk \
    libreadline6-dev \
    liblog4cplus-dev \
    libhdf5-dev \
    libncurses5-dev \
    flex \
    bison \
    libbison-dev \
    libqdbm-dev \
    make \
    rsync \
    casacore-dev \
    python3-numpy \
    python-setuptools \
    libboost-python-dev \
    libcfitsio-dev \
    wcslib-dev

# Install dependencies
RUN apt-get update && \
    apt-get install -y $DEB_SETUP_DEPENDENCIES && \
    apt-get install -y $DEB_DEPENCENDIES

# Set compiler alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GNUCOMPILER 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GNUCOMPILER 100 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-$GNUCOMPILER 100

# Setup Python virtual environment
WORKDIR /opt
RUN pip install -U pip setuptools wheel

# Get CASACORE ephem data
RUN mkdir -p /usr/share/casacore/data/ &&  \
    mkdir -p /var/lib/casacore/ &&  \
    ln -s /var/lib/casacore/data /usr/share/casacore/data
WORKDIR /usr/share/casacore/data/
RUN rsync -avz rsync://casa-rsync.nrao.edu/casa-data .

RUN pip install --no-binary python-casacore python-casacore

#####################################################################
## BUILD MAKEMS FROM SOURCE AND TEST
#####################################################################
WORKDIR /opt
RUN wget https://github.com/ska-sa/makems/archive/v1.5.4.tar.gz && \
    tar xvf v1.5.4.tar.gz && \
    rm v1.5.4.tar.gz && \
    mkdir -p /opt/makems-1.5.4/LOFAR/build/gnu_opt && \
    cd /opt/makems-1.5.4/LOFAR/build/gnu_opt && \
    cmake -DCMAKE_MODULE_PATH:PATH=/opt/makems-1.5.4/LOFAR/CMake \
          -DUSE_LOG4CPLUS=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr ../.. && \
    make -j 16 && \
    make install && \
    cd /opt/makems-1.5.4/test && \
    makems WSRT_makems.cfg && \
    rm -r /opt/makems-1.5.4



# install dsa2000_cal which is on host at ../../dsa2000_cal
WORKDIR /dsa/code
COPY dsa2000_cal dsa2000_cal
RUN pip install -r dsa2000_cal/requirements.txt && \
    pip install ./dsa2000_cal


# Prepare run
COPY services/prepare_run/src src

# Directies that will be mounted
RUN mkdir -p /dsa/data


RUN mkdir -p /dsa/run
WORKDIR /dsa/run

ENTRYPOINT [ "/bin/bash", "/dsa/code/src/entrypoint.sh" ]