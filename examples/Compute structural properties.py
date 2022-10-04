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
# # Compute structural properties
#
# Using the example of a single frame trajectory of a ZIF-4 unit cell, we'll explore the different analyses made possible by aMOF.

# %%
# Import hvplot for easy plotting
import hvplot.xarray # noqa
import hvplot.pandas # noqa
import numpy as np

# %% [markdown]
# #### Import trajectory
# Read with ase

# %%
import ase.io
atom = ase.io.read('files/ZIF-4.xyz')
traj = [atom] # single atom trajectory

# %%
atom

# %% [markdown]
# ### RDF (Radial Distribution Function)
# Each analysis is contained in a module, here `rdf`

# %%
import amof.rdf as amrdf
help(amrdf)

# %% [markdown]
# The module contains a main class used to compute the property, here 'Rdf'.

# %%
help(amrdf.Rdf)

# %% [markdown]
# This class contains a constructor to compute it from an ase trajectory: `.from_trajectory()`

# %%
rdf_object = amrdf.Rdf.from_trajectory(traj)

# %% [markdown]
# The information is always stored in an instance attribute called `.data`, which can take various forms depending on the data stored. It usually is a [pandas dataframe](https://pandas.pydata.org/) or an [xarray DataArray](https://docs.xarray.dev/en/stable/).
#
# For the RDF it's a dataframe with the position as index, and name of the atom pair as column:

# %%
rdf_object.data

# %%
rdf_object.data.hvplot(x='r', y = ['C-H', 'Zn-N'])

# %% [markdown]
# This can be stored to a file with the generic `write_to_file` method and read again from this file

# %%
rdf_object.write_to_file('filename.rdf')
new_rdf_object = amrdf.Rdf.from_file('filename.rdf')

# Do we have the same values?
np.allclose(new_rdf_object.data, rdf_object.data)

# %% [markdown]
# ### Bond-Angle distribution
# module `bad`

# %%
import amof.bad as ambad

# %%
bad_object = ambad.Bad.from_trajectory(traj, {'Zn-N': 2.5}) 

# %%
bad_object.data.hvplot(x='theta', y ="N-Zn-N", xlim = (100,120))

# %% [markdown]
# ### Coordination number
# module `cn`

# %%
import amof.cn as amcn
cn_object = amcn.CoordinationNumber.from_trajectory(traj, {'Zn-N': 2.5})
cn_object.data

# %% [markdown]
# ### Mean Squared Displacement (MSD)
# module `msd`
#
# Create a mock trajectory with random noise just to have several frames to show the MSD computations.

# %%
from copy import deepcopy
mock_traj = [deepcopy(atom)]
for i in range(10):
    atom.rattle(0.5)
    mock_traj.append(deepcopy(atom))

# %%
import amof.msd as ammsd
msd = ammsd.WindowMsd.from_trajectory(mock_traj, delta_time = 1, timestep = 1)        

# %%
msd.data.hvplot(y = ['X','Zn','C'])

# %% [markdown]
# ### Pore analysis
# module `pore`
#
# Calls Zeo++

# %%
import amof.pore as ampore
pore = ampore.Pore.from_trajectory(traj) # takes a few seconds to run

# %%
pore.data

# %% [markdown]
# ### Ring analysis
# module `ring`
#
# Here we show an example where we set-up the search only to look for imidazolates (ie cycles of C-N-C-N-C atoms).
# It takes a few (~5) minutes to run.

# %%
import amof.ring as amring
ring = amring.Ring.from_trajectory(traj, {'C-N': 1.728, 'C-C': 1.752}, max_search_depth = 6) # takes a few minutes to run

# %%
ring.data.ring.to_series()

# %%
