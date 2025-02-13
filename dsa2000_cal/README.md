# Calibration and Forward Modelling for DSA2000

This contains the `dsa2000_cal` package, which implements a library for carrying out forward modelling and
calibration of the DSA-2000. The package is predominantly written in JAX, using astropy to represent physical
quantities. The package is designed to be used with the DSA2000 radio interferometer, but can be used with others, e.g.
it is used in the OVRO-LWA project.

## Scope

In rough order of priority, the package contains the following functionality:

1. a general framework for computing visibilities for near and far emission, taking account of gravitational delays.
2. a framework for simulating systematics such as a 3D ionosphere with frozen flow, dish defects, direction-dependent
   gravitational delay errors, and smearing.
3. a framework for creating coherent synthetic sky models.
4. framework for carrying out cadenced streaming (single iteration) DD calibration.
5. the ability to perform gridding, and compute PSFs using wgridder.
6. a scalable HPC framework for carrying out streaming forward modelling, i.e. visibilities are not kept around.
7. a framework for carrying out DD calibration and subtraction in an HPC environment.

# Installation for development

This will explain how to setup a coherent development environment. This entails setuping up virtual environment, IDE,
and version control. This complete setup is recommended for any member on the team to enable a smooth development
process.

1. Install Pycharm. This is the recommended IDE for this project. You can download it from
   [here](https://www.jetbrains.com/pycharm/download/). While VSCode is also a good IDE, we like the entire team to use
   a common IDE to enable helping and especially familiarity during pair-programming sessions. Use your academic
   affiliation to install the professional version.

2Add you public SSH key to GitHub [here](https://github.com/settings/keys) so that you can clone the repository. If
you have already done this before, you can skip this.

3. Make sure you have Miniconda installed. If not you can install it with the following commands:

```bash
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
# MINICONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}
chmod +x ${MINICONDA_INSTALLER}
./${MINICONDA_INSTALLER} -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
```

4. Create conda environment called `dsa2000_cal_py` with python 3.11.

```bash
conda create -n dsa2000_cal_py python=3.11
```

5. Clone. Recommended to create a directory `/home/username/git` to store all git repositories.

```bash
cd /home/username/git
git clone git@github.com:Joshuaalbert/DSA2000-Cal.git
cd DSA2000-Cal
```

7. Install the requirements and package.

```bash
conda activate dsa2000_cal_py
pip install dsa2000_call
```

8. Set up PyCharm for development

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

9. To get access to the assets required for simulation, contact the team over gihub issues. You will need SSH access to
   a particular server.