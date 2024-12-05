# FROM python:3.11.7
FROM nvidia/cuda:12.3.0-base-ubuntu20.04

# Install Python 3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# from https://github.com/docker-library/python/blob/23c83c5a82646e123df33d056f7dfbe1d78cfc8b/3.11/bullseye/Dockerfile
# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG C.UTF-8

# runtime dependencies
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		libbluetooth-dev \
		tk-dev \
		uuid-dev \
	; \
	rm -rf /var/lib/apt/lists/*

ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION 3.11.11
ENV PYTHON_SHA256 2a9920c7a0cd236de33644ed980a13cbbc21058bfdc528febb6081575ed73be3

RUN set -eux; \
	\
	wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
	echo "$PYTHON_SHA256 *python.tar.xz" | sha256sum -c -; \
	wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
	GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
	gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
	gpg --batch --verify python.tar.xz.asc python.tar.xz; \
	gpgconf --kill all; \
	rm -rf "$GNUPGHOME" python.tar.xz.asc; \
	mkdir -p /usr/src/python; \
	tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
	rm python.tar.xz; \
	\
	cd /usr/src/python; \
	gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
	./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		--with-lto \
		--with-ensurepip \
	; \
	nproc="$(nproc)"; \
	EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
	LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-}" \
	; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
	rm python; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" \
		python \
	; \
	make install; \
	\
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
	bin="$(readlink -ve /usr/local/bin/python3)"; \
	dir="$(dirname "$bin")"; \
	mkdir -p "/usr/share/gdb/auto-load/$dir"; \
	cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
	\
	cd /; \
	rm -rf /usr/src/python; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
		\) -exec rm -rf '{}' + \
	; \
	\
	ldconfig; \
	\
	export PYTHONDONTWRITEBYTECODE=1; \
	python3 --version; \
	\
	pip3 install \
		--disable-pip-version-check \
		--no-cache-dir \
		--no-compile \
		'setuptools==65.5.1' \
		wheel \
	; \
	pip3 --version

# make some useful symlinks that are expected to exist ("/usr/local/bin/python" and friends)
RUN set -eux; \
	for src in idle3 pip3 pydoc3 python3 python3-config; do \
		dst="$(echo "$src" | tr -d 3)"; \
		[ -s "/usr/local/bin/$src" ]; \
		[ ! -e "/usr/local/bin/$dst" ]; \
		ln -svT "$src" "/usr/local/bin/$dst"; \
	done

RUN which python && which pip && python --version && pip --version

# Reset DEBIAN_FRONTEND to prevent unintended behavior in the container
ENV DEBIAN_FRONTEND=dialog

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