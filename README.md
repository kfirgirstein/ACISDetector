# ACISDetector
Detector for instruction set, architecture and compiler on raw binaries.

## jupter notebook
You can also view our results notebooks in your browser using `nbviewer` by clicking the
button below.

<a href="https://nbviewer.jupyter.org/github/vistalab-technion/cs236781-tutorials/tree/master/"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" height="50px"/></a>

## Environment set-up

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   for your platform.

2. Use conda to create a virtual environment for the assignment.
   From the assignment's root directory, run

   ```shell
   conda env create -f environment.yml
   ```

   This will install all the necessary packages into a new conda virtual environment named `cs236781`.

3. Activate the new environment by running

   ```shell
   conda activate acisdetector
   ```

4. Optionally, execute the `easy-jupter-lab.sh` script to run notebook and activate it's virtual environment.
