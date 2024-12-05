FROM python:3.11.7
ARG num_proc=5
RUN apt-get update && apt-get install -y gdb lldb sudo rsync \
    build-essential cmake g++ gcc python3 python3-pip \
    python3-numpy python3-nose python3-setuptools libcfitsio-dev \
    libhdf5-dev libboost-dev gfortran libncurses5-dev    libreadline-dev \
    flex bison libblas-dev liblapack-dev wcslib-dev \
    libfftw3-dev libhdf5-serial-dev libboost-python-dev libgsl-dev

RUN pip3 install 'numpy<2'

# install CASA
RUN git clone https://github.com/casacore/casacore.git \
    && cd casacore && mkdir build && cd build \
    && cmake -DDATA_DIR=/usr/share/casacore/data -DUSE_OPENMP=ON -DUSE_HDF5=ON -DBUILD_PYTHON3=ON -DUSE_THREADS=ON -DBoost_NO_BOOST_CMAKE=True  -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j$(num_proc) && sudo make install \
    && cd ../..

ENV CASACORE_ROOT_DIR=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN pip install python-casacore

RUN python -c "import pyrap.tables"