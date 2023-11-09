# DSA2000 Calibration and Forward Modelling

This is written as a bunch of containerised services that carry out the steps of forward modelling.

To run this, you need to have `docker` installed, if not already the case make sure your user is in the docker group (
have an admin run this for you `sudo usermod -aG docker $USER`). Then, you can run the following command:

```bash
# To run everything
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