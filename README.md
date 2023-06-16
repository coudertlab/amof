# aMOF

aMOF is a python package consisting in a collection of tools to analyze Molecular Dynamics (MD) trajectories of amorphous Metal-Organic Frameworks (MOFs).

## Functionalities

### General-purpose MD toolbox

This package brings together a number of analyses that can be performed on every MD trajectory (not necessarily MOFs), heavily using both other python packages and non-python codes.
It can compute the following properties:

- Radial Distribution Functions (RDF), based on [ASAP](https://wiki.fysik.dtu.dk/asap)
- Bond-Angle Distributions 
- Coordination numbers
- Mean Squared Displacement (MSD)
- Elastic constants from cell properties, and mechanical properties from elastic constants by calling [ELATE](https://github.com/coudertlab/elate/)
- Pore analysis by wrapping [Zeo++](http://zeoplusplus.org/), reusing code from [pysimm](https://pysimm.org/)
- Ring statistics by wrapping the [RINGS code](https://rings-code.sourceforge.net/)

The backend for manipulating trajectories is [ASE](https://wiki.fysik.dtu.dk/ase/index.html), which are [ASE trajectory](https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html) objects.

### Building units identification of aMOFs

A module called `coordination` allows the identification of the different building blocks of amorphous MOFs with an _ad hoc_ approach per MOF system.

**Only ZIF-4 is supported in the current release.**

This allows the computation of rings statistics of the metal-ligand network.

This code is designed to be compatible with [molsys](https://github.com/MOFplus/cmc-tools), and can be used to generate input files in `mfpx` format.

The detailed algorithm for ZIF-4 is presented in the supporting information of the paper [Challenges in Molecular Dynamics of Amorphous ZIFs using Reactive Force Fields](https://doi.org/10.1021/acs.jpcc.2c06305).

## Installation

### Pre-requisites

To use the `pore` and `ring` modules, [Zeo++](http://zeoplusplus.org/) and the [RINGS code](https://rings-code.sourceforge.net/) need to be installed and accessible in the system path.

First download and follow the installation instructions on their respective websites ([here for Zeo++](http://www.zeoplusplus.org/download.html) and [here for RINGS](https://rings-code.sourceforge.net/index.php?option=com_content&view=category&layout=blog&id=34&Itemid=57)).

Then for Zeo++, create the following variable on your system:
```
export ZEOpp_EXEC=/path/to/zeo++-0.3/network
```

For RINGS, ensure that the `rings` binary is in your path:
```
export PATH=$PATH:/path/to/rings/bin
```


### Installation with pip

aMOF can be installed from PyPI:
```
pip install amof
```
or directly from source:
```
git clone https://github.com/coudertlab/amof.git
cd amof
pip install . 
```

Special care should be taken with [Asap](https://wiki.fysik.dtu.dk/asap/), which can only be installed if `numpy` is already installed and is thus not a default dependency. 
To solve this, either install `asap` independently (following [their installation guide](https://wiki.fysik.dtu.dk/asap/Installation)), or first install `numpy` then install `amof` with the `rdf` [extra](https://peps.python.org/pep-0508/#extras): 
```
pip install numpy
pip install amof[rdf]
```
By default, graphical dependencies (used in the `plot` module) are not installed. To install them use the `graphics` extra:
```
pip install amof[graphics]
```

#### Installation with conda

Support for installing aMOF with `conda` is not included at the moment.
However, `conda` can be used to first install all dependencies before installing `amof` with `pip`.
```
conda install -c conda-forge ase=3.20.1 asap3 pandas numpy xarray=0.19.0 dask netCDF4 bottleneck ase asap3 joblib pymatgen hvplot atomman jupyter jupyterlab jupytext matplotlib networkx scipy plotly seaborn cairosvg pyarrow selenium
```

### Documentation

A documentation is can easily be created with [pdoc](https://pdoc3.github.io/pdoc/). 

Simply run (with pdoc installed):
```
pdoc --html --output-dir path\to\docs path\to\amof
```

## Usage

### Examples

Examples analyses can be found in the `examples` folder of this repository.

### Citation

If you use the python package in published results (paper, conference, etc.), please cite the first paper for which it was developed: [Challenges in Molecular Dynamics of Amorphous ZIFs using Reactive Force Fields](https://doi.org/10.1021/acs.jpcc.2c06305).