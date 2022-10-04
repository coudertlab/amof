# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Elastic constants and mechanical properties
#
# Using a toy example of cell information stored during a [CP2K](https://www.cp2k.org/) MD run, we'll extract the elastic constants using the strain-fluctuation method before computing the resulting mechanical properties using [ELATE](https://github.com/coudertlab/elate/).
#
# Please note that due to the really small size of the cell information provided, Elastic Constant are not converged and the results unphysical.

# %%
import numpy as np

# %% [markdown]
# ### Prepare cell information
# Cell information can be stored under a variety of format depending on the MD scheme used.
#
# Here we show the example of the cell information formatted as a CP2K output, as aMOF contains a function to read it straight away, but any format of cell is possible as long as it end up formatted as required by aMOF (see below).

# %%
from amof.files.cp2k import read_tabular
cell_output = read_tabular('files/toy_trajectory.cell')

# %%
cell_output = cell_output[0:10000]

# %%
cell_output[0:5]

# %%
# Format as list of (a_x    a_y   a_z      b_x     b_y    b_z   c_x      c_y     c_z)
cell = np.array(cell_output.drop(columns=['Time', 'Volume']))
# Then reshape as list of ((a_x    a_y   a_z),
#                             (b_x     b_y    b_z),
#                               (c_x      c_y     c_z))
cell = cell.reshape(len(cell_output),3,3)

# %%
cell

# %% [markdown]
# ### Elastic Constants
# The module `elastic` contains the `ElasticConstant` class that can load the cell information under any format that ASE [.set_cell()](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.set_cell) method can process

# %%
import amof.elastic as amela
help(amela.ElasticConstant.from_cell)

# %%
elastic_constant = amela.ElasticConstant.from_cell(cell, 300, final_value = True)

# %%
C = elastic_constant.Cmat
C

# %%
amela.print_Cmat(C)

# %% [markdown]
# ### Mechanical properties
# aMOF includes a wrapper to call [ELATE](https://github.com/coudertlab/elate/) to extract average mechanical properties through the `MechanicalProperties` class.

# %%
help(amela.MechanicalProperties.from_elastic)

# %%
C_as_list = np.array(C).tolist() # Convert to ELATE input
mech = amela.MechanicalProperties.from_elastic(C_as_list)

# %%
mech.data
