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
# # Identify building units of ZIF-4
#
# Using the example of a single frame trajectory of a ZIF-4 crystal unit cell, we'll explore the building units identification and subsequent ring statistics computation.
#
# The module is `coordination`

# %%
# Import hvplot for easy plotting
import hvplot.xarray # noqa
# import hvplot.pandas # noqa
import numpy as np

# %% [markdown]
# #### Import trajectory
# Read with ase

# %%
import ase.io
atom = ase.io.read('files/ZIF-4.xyz')
traj = [atom] # single atom trajectory

# %%
traj[0].get_global_number_of_atoms()

# %% [markdown]
# ### Reduced trajectory
# First we create an reduced trajectory by identifying the building units for each frame of the trajectory and saving only the positions of these buildings units.

# %%
import amof.coordination.reduce as amred
help(amred)

# %%
red_traj = amred.reduce_trajectory(traj, 'ZIF-4')

# %% [markdown]
# This lead to the creation of a `ReducedTrajectory` object which contains a `trajectory` attribute with the reduced trajectory.

# %%
red_traj.trajectory[0].get_global_number_of_atoms()

# %% [markdown]
# It also contains a `report_search` attribute with the log of how the search for building units went

# %%
red_traj.report_search

# %% [markdown]
# ### Ring analysis
# Ring statistics can be computed from a reduced trajectory, which allows the computation of rings statistics of the metal-ligand network.

# %%
import amof.ring as amring
ring = amring.Ring.from_reduced_trajectory(red_traj, max_search_depth = 16)

# %%
ring.data.ring.hvplot.bar(x = 'ring_size')

# %%
