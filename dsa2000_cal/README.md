# Calibration and Forward Modelling for DSA2000

This contains the `dsa2000_cal` package, which implements a library for carrying out forward modelling and
calibration of radio interferometers. The package is predominantly written in JAX, using astropy to represent physical
quantities. The package is designed to be used with the DSA2000 radio interferometer, but can be used with others, e.g.
it is used in the OVRO-LWA project.

## Scope

In rough order of priority, the package contains the following functionality:

1. a general framework for computing visibilities for numerous types of source, in both far- and near-field, using VLBI
   precise delays, including: WSClean points and Gaussian component lists sources, general sky models defined by FITS in
   SIN projection, and RFI emitters defined by physical location, velocity, and (potentially temporally varying)
   auto-correlation spectrum.
2. a framework for simulating systematics including per-antenna beams, a 3D ionosphere with frozen flow, a comprehensive
   set of dish effects, direction-dependent gravitational delay errors, and smearing.
3. a framework for creating coherent synthetic sky models spanning multiple pointings, e.g. ability to create moasics
   with coherent source models spanning pointings. This includes: a-team sources, 100 compact 10Jy (L-band) sources
   scattered over the sky, T-RECS faint sources, injected Illustris galaxies placed at different redshifts, other
   synthetic point and Gaussian sources injected within the FoV.
4. framework for carrying out cadenced streaming (single iteration) DD calibration.
5. the ability to perform gridding, and compute PSFs using wgridder.
6. a scalable HPC framework for carrying out streaming forward modelling of: simulation of systematics -> simulation of
   visibilities -> flagging -> calibration -> imaging, without storing visibilties.
7. modern Python friendly self-contained reduced MS definition, using `pydantic` for validation, `astropy` for
   physical quantities, and `hdf5` for storage.
8. low-level interfaces for streaming generators of visibility data to perform streaming calibration and subtraction
   using PubSub style of coordination over socket, and shared memory passthrough of data.

# Installation for development

This will explain how to setup a coherent development environment. This entails setuping up virtual environment, IDE,
and version control. This complete setup is recommended for any member on the team to enable a smooth development
process.

1. Install Pycharm. This is the recommended IDE for this project. You can download it from
   [here](https://www.jetbrains.com/pycharm/download/). While VSCode is also a good IDE, we like the entire team to use
   a common IDE to enable helping and especially familiarity during pair-programming sessions. Use your academic
   affiliation to install the professional version.
2. Install Git and Git LFS.

```bash
sudo apt-get install git git-lfs
```

3. Add you public SSH key to GitHub [here](https://github.com/settings/keys) so that you can clone the repository. If
   you have already done this before, you can skip this.

4. Make sure you have Miniconda installed. If not you can install it with the following commands:

```bash
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
# MINICONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}
chmod +x ${MINICONDA_INSTALLER}
./${MINICONDA_INSTALLER} -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
```

5. Create conda environment called `dsa2000_cal_py` with python 3.11.

```bash
conda create -n dsa2000_cal_py python=3.11
```

6. Clone and install git LFS. Recommended to create a directory `/home/username/git` to store all
   git repositories.

```bash
cd /home/username/git
git clone git@github.com:Joshuaalbert/DSA2000-Cal.git
cd DSA2000-Cal
git lfs install
```

7. Install the requirements and package.

```bash
cd dsa2000_cal
conda activate dsa2000_cal_py
pip install -r requirements.txt
pip install -r requirements-notebooks.txt
pip install -r requirements-tests.txt
pip install .
```

7. Set up PyCharm for development

    1. Make sure you have created a `dsa2000_cal_py` conda env as above, and installed requirements files.
    2. Create a new project in PyCharm in the repo root directory `/home/username/git/DSA2000-Cal`. Use an empty project
       name to force it to use existing content as project source.
    4. Open the **project settings > Tools > Python Integrated Tools**. Choose `pytest` as the default test runner.
       And `Google`as docstring format.
    5. Open the **project settings > Project Interpreter**. Add a new Conda interpreter and select from the list the
       conda
       env you created above, `dsa2000_cal_py`.
    5. Open the **project settings > Project Structure**. Add `/home/username/git/DSA2000-Cal/dsa2000_cal` as a source
       folder. This will allow the IDE to use your live source code for code completion and running tests.

## Using Git LFS for large files required for the project

Several largish files are required and stored using Git LFS. To track a large file use,

```bash
# To track a large file use
git lfs track "path/to/file"
```

after which you can use normal git commands to commit and push, and the large files are handled in the backround.
Confirm with the team **before** committing new large files. For this to work you need to have done `git lfs install`
after cloning, as instructed above.