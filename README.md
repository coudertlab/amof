# aMOF

aMOF is a collection of tools to analyze Molecular Dynamics (MD) trajectories of amorphous Metal-Organic Frameworks (MOFs).

## Functionalities

### General-purpose MD toolbox

This package brings together a number of analysis that can be performed on every MD trajectory (not necessarily MOFs), heavily using pre-existing python packages or non-python codes.
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

The detailed algorithm for ZIF-4 is presented in the supporting information of the working paper [Challenges in Molecular Dynamics of Amorphous ZIFs using Reactive Force Fields](https://doi.org/10.26434/chemrxiv-2022-lw5n8).

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


### Python package
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

### Documentation

A documentation is available on [xxx]() and can easily be created with [pdoc](https://pdoc3.github.io/pdoc/). 

Simply run (with pdoc installed):
```
pdoc --html --output-dir path\to\docs path\to\amof
```

## Usage

Examples analyses can be found in the `examples` folder of this repository.