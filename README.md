# DSA2000 Calibration and Forward Modelling

This repo contains two components:

1. The [dsa2000_cal package](dsa2000_cal), which implements a library for carrying out forward modelling and
   calibration of radio interferometers. The package is predominantly written in JAX, using astropy to represent
   physical
   quantities. The package is designed to be used with the DSA2000 radio interferometer, but can be used with others,
   e.g.
   it is used in the OVRO-LWA project. See [the package README](dsa2000_cal/README.md) for more details.
2. [Workflows](workflows) which contains scripts and Dockerfiles for running distributed workflows utilising the
   `dsa2000_cal` package. This includes setting up a distributed streaming calibration with low-level interface exposed,
   sharing data between containers through sharded memory and common namespace; streaming forward modelling to simulate
   observations; as well as non-streaing variants of the above two workflows.

# Deployment

To run any workflows, you need to have `docker` installed, if not already the case make sure your user is in the docker
group (
have an admin run this for you `sudo usermod -aG docker $USER`). Then, you can run the following command in a given
workflow folder:

```bash
./run.sh
```

We use Git LFS to store the data, so you will need to have that installed. You can install it with the following
commands:

```bash
sudo apt-get install git-lfs
# Then inside repo directory
git lfs install
# To track a large file use
git lfs track "path/to/file"
```

## To run with GPU support, you need to have the Nvidia driver and `nvidia-docker` installed.

If you have the Nvidia driver installed then you should be able to run:

```bash
nvidia-smi
```

and see something like:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
...
```

On Ubuntu, you can do this with the following commands:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

On CentOS/RHEL/Fedora, you can do this with the following commands:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
sudo rpm --import https://nvidia.github.io/nvidia-docker/gpgkey
sudo curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo -o /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker\
```

## TODO: set polarisations in Beam 