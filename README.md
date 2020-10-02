# ACISDetector

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/badge/version-v1.0-blue)](https://github.com/kfirgirstein/ACISDetector/releases)

![ACIS Logo](/jupyter_utils/ACIS_Logo.png)


Detector for instruction set, architecture and compiler on raw binaries.


## Installation

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   for your platform.

2. Use conda to create a virtual environment for the assignment.
   From the assignment's root directory, run

   ```shell
   conda env create -f environment.yml
   ```

   This will install all the necessary packages into a new conda virtual environment named `acisdetector`.

3. Activate the new environment by running

   ```shell
   conda activate acisdetector
   ```

4. Optionally, execute the `easy-jupter-lab.sh` script to run notebook and activate it's virtual environment.

#### jupyter notebook
You can also view our results notebooks in your browser using `jupter notebook`!

   ```shell
   juypter lab
   ```
##### Notes:
- On Windows, you should also install Microsoft's [Build Tools for Visual
  Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)
  before installing the conda env.  Make sure "C++ Build Tools" is selected during installation.
- After a new tutorial is added, you should run `conda env update` from the repo
  directory to update your dependencies since each new tutorial might add ones.
 
 
 ## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. We recommend to follow [Contribute Guideline](/CONTRIBUTING.md)

## License
[MIT Licensed](https://choosealicense.com/licenses/mit/) (file [LICENSE](/LICENSE)).© [Kfir Girstein](https://github.com/kfirgirstein/) & [yhonatanlus](https://github.com/Yehonatanlus), 2020
