# STEAK


<img src="figs/steakLogo.png" width="250">

## How to run STEAK

Need to install the anaconda or miniconda environment. You can download it by following the link below:

https://www.anaconda.com/download

https://docs.anaconda.com/miniconda/install/

Install all libraries that you need:

`conda env create -f stenv.yml`

then : 

`conda activate stenv`


A set of default parameters is already given in (all default parameters can be modifiable):

`settings/parameters.py`

:warning: The entire simulation uses the anode geometry parameters given in this file (field calculation, electron drift and field response calculation).

The geometry of the readout plane is given in:

`settings/geometry.py`

:warning: This simulation is hardcoded to be performed with the perforated plane readout technology (can't be used for the wire chambers).

The input agreement values for simulation can be modified in:
`settings/args.py`

## Field STEAK

To launch <b>drift field calculation</b>, type `python drift.py` with the following arguments:<br/>

* `-conv` give a stopping criterion for the finite difference method <br/>, (default value 0.1 V)
* `-namefile` give a name for the output file, (default name 'drift') 
* '-
