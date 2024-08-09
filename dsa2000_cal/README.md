# Calibration and Forward Modelling for DSA2000

# Installation

## Install miniconda

```bash
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
# MINICONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}
chmod +x ${MINICONDA_INSTALLER}
./${MINICONDA_INSTALLER} -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
```

## Create conda environment

```bash
conda create -n dsa2000_cal_py python=3.11
```

## Install package

```bash
conda activate dsa2000_cal_py
pip install -r requirements.txt
pip install -r requirements-notebooks.txt
pip install -r requirements-tests.txt
pip install .
```

# Setting up PyCharm for development

1. Create a conda env as above. Install requirements files. You do not need to install the package.
2. Create a new project in PyCharm in the repo root directory, with a blank project name, selecting to use existing
   sources.
4. Open the project settings > Tools > Python Integrated Tools. Choose `pytest` as the default test runner. And `Google`
   as docstring format.
5. Open the project settings > Project Interpreter. Add the conda env you created above.
5. Open the project settings > Project Structure. Add `/path/to/repo/dsa2000_cal` as a source root.

