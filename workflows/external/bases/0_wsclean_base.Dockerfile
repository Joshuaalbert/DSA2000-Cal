# Using the same base image
ARG NVIDIA_VERSION=11.8.0
#11.3.1

FROM nvidia/cuda:${NVIDIA_VERSION}-devel-ubuntu22.04

# Copy sources list and modify it


# Set some environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_PRIORITY critical
ENV GNUCOMPILER 10
ENV DEB_SETUP_DEPENDENCIES \
    dpkg-dev \
    g++-$GNUCOMPILER \
    gcc-$GNUCOMPILER \
    libc-dev \
    cmake \
    gfortran-$GNUCOMPILER \
    git \
    wget \
    ninja-build \
    file

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
    libncurses5-dev \
    flex \
    bison \
    libbison-dev \
    libqdbm-dev \
    make \
    rsync \
    python3-numpy \
    python-setuptools \
    libboost-python-dev \
    libblas-dev  \
    build-essential  libarmadillo-dev libgsl-dev \
    libboost-filesystem-dev libboost-system-dev libboost-date-time-dev \
    libboost-program-options-dev libboost-test-dev \
    libxml2-dev \
    libgtkmm-3.0-dev \
    libboost-numpy-dev liblua5.3-dev \
    casacore-dev casacore-tools pybind11-dev \
    liblapacke-dev \
    python3
#    libboost-test1.71.0


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
## BUILD IDG
#####################################################################

WORKDIR /opt

RUN git clone https://git.astron.nl/RD/idg.git

RUN cd /opt/idg && \
    # Unshallowing is needed for determining the IDG version.
    if $(git rev-parse --is-shallow-repository); then git fetch --unshallow; fi && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/usr \
      "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/compat;/usr/local/cuda/lib64" \
      "-DCMAKE_CXX_FLAGS=-isystem /usr/local/cuda/include" \
      -DBUILD_LIB_CUDA=On -DBUILD_PACKAGES=On -DBUILD_WITH_PYTHON=ON -DBUILD_LIB_CPU=On -DBUILD_LIB_GPU=On -DBUILD_LIB_CUDA=On && \
    ninja package

# -DBUILD_TESTING=On


# Installing Boost in Ubuntu 20.04 requires the version number.
#ARG BOOST_VERSION=1.71.0

# The metadata should include the IDG version.

RUN cp -r /opt/idg/build/idg-*.deb /opt/idg/
RUN cp -r /opt/idg/build/bin /opt/idg/

RUN apt-get update && \
#    apt-get install -y --no-install-recommends libboost-test${BOOST_VERSION} && \
    apt install -y /opt/idg/*.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

###
## EveryBeam
###
ENV EVERYBEAM_VERSION_TAG v0.4.0

WORKDIR /opt

RUN mkdir /opt/everybeam && cd /opt/everybeam \
    && git clone https://git.astron.nl/RD/EveryBeam.git src \
    && (cd src && git checkout $EVERYBEAM_VERSION_TAG) \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_WITH_PYTHON=ON ../src -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && make install -j4 \
    && cd / && rm -rf everybeam


#####################################################################
## BUILD WSCLEAN
#####################################################################
WORKDIR /opt
RUN  mkdir /opt/wsclean && cd /opt/wsclean && \
     git clone --depth 1 --branch v3.3 https://gitlab.com/aroffringa/wsclean.git src \
     && mkdir build && cd build && \
     cmake -DCMAKE_PREFIX_PATH=/opt/idg -DCMAKE_INSTALL_PREFIX=/usr ../src && \
     make install -j4
