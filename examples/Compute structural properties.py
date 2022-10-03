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
# Using the example of a single frame trajectory of a ZIF-4 unit cell, we'll explore the different analyses made possible by aMOF

# %%
# Import hvplot for easy plotting
# import hvplot.xarray # noqa
import hvplot.pandas # noqa
# import xarray as xr
# import numpy as np

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
# ### RDF

# %%
import amof.rdf as amrdf
rdf = amrdf.Rdf.from_trajectory(traj)

# %%
rdf.rdf_data.hvplot(x='r', y = ['C-H', 'Zn-N'])

# %% [markdown]
# ### Bond-Angle distribution

# %%
import amof.bad as ambad
help(ambad.Bad.from_trajectory)

# %%
rmc = ambad.Bad.from_trajectory(traj, {'Zn-N': 2.5}) 

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
abinitio = xr.open_dataset(path_to_gaillac_dataset+'bad.comp.nc')
# .sel(run_id = ['ZIF4_15glass01', 'ZIF4_15glass02', 'ZIF4_15glass07']).bad

# %%
abinitio['phase'] = abinitio.phase.astype('str')

# %%
abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').bad

# %%
abinitio = abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').bad

# %%
avgglass = abinitio.sel(phase='glass').mean('run_id_level_0')
ZIF4_15glass01 = abinitio.sel(run_id_level_0 = 'ZIF4_15glass01')
crystal = abinitio.sel(phase='crystal').squeeze('run_id_level_0')
everyglass = abinitio.sel(phase='glass').rename({"run_id_level_0":"run_id"})

# %%
prop = "N-Zn-N"
plot = (
       everyglass.sel(atom_triple=prop).hvplot(x='theta', label='ab initio glass', by = 'run_id') * 
    avgglass.sel(atom_triple=prop).hvplot(x='theta', label='average glass')

       )
# save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
prop = "N-Zn-N"
plot = (rmc.bad_data.rolling(100, center = True).mean().hvplot(x='theta', y =prop, label='RMC glass') * 
 da.bad.sel(system='reaxff_amorphous', atom_triple = prop).hvplot(x = 'theta', kind='line', 
    ylabel='Bond Angle', xlabel='theta (deg)', label='ReaxFF glass',
                        title = f'Comparison of models in their native environment - {prop}') *
       avgglass.sel(atom_triple=prop).hvplot(x='theta', label='ab initio glass (average)') *
        crystal.sel(atom_triple=prop).hvplot(x='theta', label='Crystal (ab initio, 300K)')
       )
# save_hvplot(plot, f'Model_comparison_bad_{prop}')
plot

# %% [markdown]
# ### RDF

# %%
import amof.rdf as amrdf
rdf = amrdf.Rdf.from_trajectory(traj)

# %%
rdf.rdf_data.hvplot(x='r', y="Zn-N")

# %%
import amof.rdf as srdf
rmc = srdf.Rdf.from_trajectory(traj)

# %%
# rmc.rdf_data['Zn-N'] = rmc.rdf_data['Zn-N'] * np.sqrt(34)

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
abinitio = xr.open_dataset(path_to_gaillac_dataset+'rdf.comp.nc')
# .sel(run_id = ['ZIF4_15glass01', 'ZIF4_15glass02', 'ZIF4_15glass07']).bad

# %%
abinitio = abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').rdf

# %%
avgglass = abinitio.sel(phase='glass').mean('run_id_level_0')
ZIF4_15glass01 = abinitio.sel(run_id_level_0 = 'ZIF4_15glass01')
crystal = abinitio.sel(phase='crystal').squeeze('run_id_level_0')
everyglass = abinitio.sel(phase='glass').rename({"run_id_level_0":"run_id"})

# %%
prop = "Zn-N"
plot = (
       everyglass.sel(atom_pair=prop).hvplot(x='r', label='ab initio glass', by = 'run_id') * 
    avgglass.sel(atom_pair=prop).hvplot(x='r', label='average glass')

       )
# save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
rdf.rdf_data.hvplot(x='r', y=atom_pair, label='RMC initial file')

# %%
import amof.bad as sbad
# bad = sbad.Bad.from_trajectory(traj, {'Zn-N': 2.5})
# bad.write_to_file(path_to_rmc_prop / 'bad')

bad = sbad.Bad.from_file(path_to_rmc_prop / 'bad')

# %%
# bad.bad_data.hvplot(x='theta', y ='N-Zn-N')
bad.bad_data.rolling(50, center = True).mean().hvplot(x='theta', y ='N-Zn-N')

# %%
prop = "N-Zn-N"
plot = (bad.bad_data.rolling(100, center = True).mean().hvplot(x='theta', y =prop, label='RMC initial file') * 
 da.bad.sel(system=system, atom_triple = prop).hvplot(x = 'theta', kind='line', 
    ylabel='Bond Angle', xlabel='theta (deg)', label='ReaxFF simulation', title = 'RMC glass - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
atom_pair = "Zn-N"
xmin, xmax = 1.5, 6
plot = (rdf.rdf_data.rolling(5, center =True).mean().hvplot(x='r', y =atom_pair, label='RMC initial file') * 
    da.rdf.sel(atom_pair = atom_pair, system=system).hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', title = 'RMC glass - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_RMC_rdf_{atom_pair}')
plot

# %%
atom_pair = "Zn-Zn"
xmin, xmax = 3, 8.5
plot = (rdf.rdf_data.rolling(10, center =True).mean().hvplot(x='r', y =atom_pair, label='RMC initial file') * 
    da.rdf.sel(atom_pair = atom_pair, system='rmc_bennet').hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', 
       title = 'RMC glass - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_RMC_rdf_{atom_pair}')
plot

# %%
import amof.cn as scn
# cn = scn.CoordinationNumber.from_trajectory(traj, {'Zn-N': 2.5})
# cn.write_to_file(path_to_rmc_prop / 'cn')

cn = scn.CoordinationNumber.from_file(path_to_rmc_prop / 'cn')

# %%
prop = "Zn-N"
print('RMC glass - Introduction in ReaxFF simulation')
print('Coordination number comparison for ', prop)
print("RMC initial file: ", "{:.2f}".format(cn.cn_data[prop][0]))
print("ReaxFF:           ", "{:.2f}".format(da.coordination_number.sel(atom_pair = prop, system='rmc_bennet').dropna('Step').mean('Step').item()))

# %%
import amof.pore as spore
# pore = spore.Pore.from_trajectory(traj)
# pore.write_to_file(path_to_rmc_prop / 'pore')

pore = spore.Pore.from_file(path_to_rmc_prop / 'pore')

# %%
pore.surface_volume[['AV_cm^3/g', 'NAV_cm^3/g']]

# %%
print('RMC glass - Introduction in ReaxFF simulation')
print('Pore volume comparison ')
print('             AV_cm^3/kg  NAV_cm^3/g')
print("RMC initial file: ", "{:.2f}".format(1000*pore.surface_volume['AV_cm^3/g'][0]), 
      "{:.2f}".format(1000*pore.surface_volume['NAV_cm^3/g'][0]))
print("ReaxFF:           ", 
      "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'AV_cm^3/g').dropna('Step').mean('Step').item()),
           "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'NAV_cm^3/g').dropna('Step').mean('Step').item()),
)

# %%

# %% [markdown]
# ### ZIF4_15glass01

# %%
system = 'abinitio_glass'

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
bad = xr.open_dataset(path_to_gaillac_dataset+'bad.comp.nc').sel(run_id = 'ZIF4_15glass01').bad

# %%
bad

# %%
prop = "N-Zn-N"
plot = (bad.sel(atom_triple=prop).hvplot(x='theta', label='ab initio simulation') * 
 da.bad.sel(system=system, atom_triple = prop).hvplot(x = 'theta', kind='line', 
    ylabel='Bond Angle', xlabel='theta (deg)', label='ReaxFF simulation', title = 'ab initio glass - Introduction in ReaxFF simulation'))
# save_hvplot(plot, f'ReaxFFinfluence_{system}_bad_{prop}')
plot

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
rdf = xr.open_dataset(path_to_gaillac_dataset+'rdf.comp.nc').sel(run_id = 'ZIF4_15glass01').rdf

# %%
rdf.sel(atom_pair = 'Zn-N').hvplot(x='r', label='RMC initial file')

# %%
atom_pair = "Zn-N"
xmin, xmax = 1.5, 6
plot = (rdf.sel(atom_pair = atom_pair).hvplot(x='r', label='ab initio simulation') * 
    da.rdf.sel(atom_pair = atom_pair, system=system).hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', title = 'ab initio glass - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_{system}_rdf_{atom_pair}')
plot

# %%
atom_pair = "Zn-Zn"
xmin, xmax = 3, 8.5
plot = (rdf.sel(atom_pair = atom_pair).hvplot(x='r', label='ab initio simulation') * 
    da.rdf.sel(atom_pair = atom_pair, system=system).hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', title = 'ab initio glass - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_{system}_rdf_{atom_pair}')
plot

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
cn = xr.open_dataset(path_to_gaillac_dataset+'cn.comp.nc').sel(run_id = 'ZIF4_15glass01').coordination_number

# %%
cn.hvplot(x = 'Step', y = 'coordination_number', kind = 'line', dynamic = True)

# %%
prop = "Zn-N"
print('ab initio glass - Introduction in ReaxFF simulation')
print('Coordination number comparison for ', prop)
print("ab initio: ", "{:.2f}".format(cn.sel(atom_pair = prop).dropna('Step').mean('Step').item()))
print("ReaxFF:    ", "{:.2f}".format(da.coordination_number.sel(atom_pair = prop, system=system).dropna('Step').mean('Step').item()))

# %%
pore = xr.open_dataset(path_to_gaillac_dataset+'pore.comp.nc').sel(run_id = 'ZIF4_15glass01').pore

# %%
pore.surface_volume[['AV_cm^3/g', 'NAV_cm^3/g']]

# %%
print('ab initio glass - Introduction in ReaxFF simulation')
print('Pore volume comparison ')
print('             AV_cm^3/kg  NAV_cm^3/g')
print("ab initio:           ", 
      "{:.2f}".format(1000*pore.sel(pore_var = 'AV_cm^3/g').dropna('Step').mean('Step').item()),
           "{:.2f}".format(1000*pore.sel(pore_var = 'NAV_cm^3/g').dropna('Step').mean('Step').item()),
)
print("ReaxFF   :           ", 
      "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'AV_cm^3/g').dropna('Step').mean('Step').item()),
           "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'NAV_cm^3/g').dropna('Step').mean('Step').item()),
)

# %% [markdown]
# ### crystal

# %%
system = 'crystallographic_crystal'
title = 'Crystal'
gaillac_id = 'ZIF4_300K'

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
bad = xr.open_dataset(path_to_gaillac_dataset+'bad.comp.nc').sel(run_id = gaillac_id).bad

# %%
bad

# %%
prop = "N-Zn-N"
plot = (bad.sel(atom_triple=prop).hvplot(x='theta', label='ab initio simulation') * 
 da.bad.sel(system=system, atom_triple = prop).hvplot(x = 'theta', kind='line', 
    ylabel='Bond Angle', xlabel='theta (deg)', label='ReaxFF simulation', title = f'{title} - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_{system}_bad_{prop}')
plot

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
rdf = xr.open_dataset(path_to_gaillac_dataset+'rdf.comp.nc').sel(run_id = gaillac_id).rdf

# %%
atom_pair = "Zn-N"
xmin, xmax = 1.5, 6
plot = (rdf.sel(atom_pair = atom_pair).hvplot(x='r', label='ab initio simulation') * 
    da.rdf.sel(atom_pair = atom_pair, system=system).hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', title = f'{title} - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_{system}_rdf_{atom_pair}')
plot

# %%
atom_pair = "Zn-Zn"
xmin, xmax = 3, 8.5
plot = (rdf.sel(atom_pair = atom_pair).hvplot(x='r', label='ab initio simulation') * 
    da.rdf.sel(atom_pair = atom_pair, system=system).hvplot.line(x = 'r', 
                        xlim=(xmin, xmax),
   ylabel='Radial Distribution Function', xlabel='r (Ang)', label='ReaxFF simulation', title = f'{title} - Introduction in ReaxFF simulation'))
save_hvplot(plot, f'ReaxFFinfluence_{system}_rdf_{atom_pair}')
plot

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
cn = xr.open_dataset(path_to_gaillac_dataset+'cn.comp.nc').sel(run_id = gaillac_id).coordination_number

# %%
cn.hvplot(x = 'Step', y = 'coordination_number', kind = 'line', dynamic = True)

# %%
prop = "Zn-N"
print(f'{title} - Introduction in ReaxFF simulation')
print('Coordination number comparison for ', prop)
print("ab initio: ", "{:.2f}".format(cn.sel(atom_pair = prop).dropna('Step').mean('Step').item()))
print("ReaxFF:    ", "{:.2f}".format(da.coordination_number.sel(atom_pair = prop, system=system).dropna('Step').mean('Step').item()))

# %%
pore = xr.open_dataset(path_to_gaillac_dataset+'pore.comp.nc').sel(run_id = gaillac_id).pore

# %%
pore.sel(pore_var = 'AV_cm^3/g').dropna('Step').hvplot(kind='hist')

# %%
print(f'{title} - Introduction in ReaxFF simulation')
print('Pore volume comparison ')
print('             AV_cm^3/kg  NAV_cm^3/g')
print("ab initio:           ", 
      "{:.2f}".format(1000*pore.sel(pore_var = 'AV_cm^3/g').dropna('Step').mean('Step').item()),
           "{:.2f}".format(1000*pore.sel(pore_var = 'NAV_cm^3/g').dropna('Step').mean('Step').item()),
)
print("ReaxFF   :           ", 
      "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'AV_cm^3/g').dropna('Step').mean('Step').item()),
           "{:.2f}".format(1000*da.pore.sel(system = system, pore_var = 'NAV_cm^3/g').dropna('Step').mean('Step').item()),
)

# %% [markdown]
# ## Compare different systems in their native environment

# %% [markdown]
# ### bad

# %%
import ase.io
import hvplot.pandas # noqa
traj = [ase.io.read('/home/nicolas/Documents/Projects/glasses/Other_groups/Bennet/Gaillac2017/mqgzif4_ase.cif')]

# %%
import amof.bad as sbad
rmc = sbad.Bad.from_trajectory(traj, {'Zn-N': 2.5})

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
abinitio = xr.open_dataset(path_to_gaillac_dataset+'bad.comp.nc')
# .sel(run_id = ['ZIF4_15glass01', 'ZIF4_15glass02', 'ZIF4_15glass07']).bad

# %%
abinitio['phase'] = abinitio.phase.astype('str')

# %%
abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').bad

# %%
abinitio = abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').bad

# %%
avgglass = abinitio.sel(phase='glass').mean('run_id_level_0')
ZIF4_15glass01 = abinitio.sel(run_id_level_0 = 'ZIF4_15glass01')
crystal = abinitio.sel(phase='crystal').squeeze('run_id_level_0')
everyglass = abinitio.sel(phase='glass').rename({"run_id_level_0":"run_id"})

# %%
prop = "N-Zn-N"
plot = (
       everyglass.sel(atom_triple=prop).hvplot(x='theta', label='ab initio glass', by = 'run_id') * 
    avgglass.sel(atom_triple=prop).hvplot(x='theta', label='average glass')

       )
# save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
prop = "N-Zn-N"
plot = (rmc.bad_data.rolling(100, center = True).mean().hvplot(x='theta', y =prop, label='RMC glass') * 
 da.bad.sel(system='reaxff_amorphous', atom_triple = prop).hvplot(x = 'theta', kind='line', 
    ylabel='Bond Angle', xlabel='theta (deg)', label='ReaxFF glass',
                        title = f'Comparison of models in their native environment - {prop}') *
       avgglass.sel(atom_triple=prop).hvplot(x='theta', label='ab initio glass (average)') *
        crystal.sel(atom_triple=prop).hvplot(x='theta', label='Crystal (ab initio, 300K)')
       )
# save_hvplot(plot, f'Model_comparison_bad_{prop}')
plot

# %% [markdown]
# ### rdf

# %%
import ase.io
import hvplot.pandas # noqa
traj = [ase.io.read('/home/nicolas/Documents/Projects/glasses/Other_groups/Bennet/Gaillac2017/mqgzif4_ase.cif')]

# %%
import amof.rdf as srdf
rmc = srdf.Rdf.from_trajectory(traj)

# %%
# rmc.rdf_data['Zn-N'] = rmc.rdf_data['Zn-N'] * np.sqrt(34)

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
abinitio = xr.open_dataset(path_to_gaillac_dataset+'rdf.comp.nc')
# .sel(run_id = ['ZIF4_15glass01', 'ZIF4_15glass02', 'ZIF4_15glass07']).bad

# %%
abinitio = abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').rdf

# %%
avgglass = abinitio.sel(phase='glass').mean('run_id_level_0')
ZIF4_15glass01 = abinitio.sel(run_id_level_0 = 'ZIF4_15glass01')
crystal = abinitio.sel(phase='crystal').squeeze('run_id_level_0')
everyglass = abinitio.sel(phase='glass').rename({"run_id_level_0":"run_id"})

# %%
prop = "Zn-N"
plot = (
       everyglass.sel(atom_pair=prop).hvplot(x='r', label='ab initio glass', by = 'run_id') * 
    avgglass.sel(atom_pair=prop).hvplot(x='r', label='average glass')

       )
# save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
prop = "Zn-N"
xmin, xmax = 1.5, 6
plot = (rmc.rdf_data.rolling(5, center = True).mean().hvplot(x='r', y =prop, label='RMC glass') * 
 da.rdf.sel(system='reaxff_amorphous', atom_pair=prop).hvplot(x = 'r', kind='line', 
    ylabel='RDF', xlabel='r (Ang)', label='ReaxFF glass', 
                title = f'Comparison of models in their native environment - {prop}', xlim=(xmin, xmax)) *
       avgglass.sel(atom_pair=prop).hvplot(x='r', label='ab initio glass (average)') *
        crystal.sel(atom_pair=prop).hvplot(x='r', label='Crystal (ab initio, 300K)')
       )
save_hvplot(plot, f'Model_comparison_rdf_{prop}')
plot

# %%
prop = "Zn-Zn"
xmin, xmax = 3, 8
plot = (rmc.rdf_data.rolling(10, center = True).mean().hvplot(x='r', y =prop, label='RMC glass') * 
 da.rdf.sel(system='reaxff_amorphous', atom_pair=prop).hvplot(x = 'r', kind='line', 
    ylabel='RDF', xlabel='r (Ang)', label='ReaxFF glass', 
                title = f'Comparison of models in their native environment - {prop}', xlim=(xmin, xmax)) *
       avgglass.sel(atom_pair=prop).hvplot(x='r', label='ab initio glass (average)') *
        crystal.sel(atom_pair=prop).hvplot(x='r', label='Crystal (ab initio, 300K)')
       )
save_hvplot(plot, f'Model_comparison_rdf_{prop}')
plot

# %%

# %% [markdown]
# ### cn

# %%
import ase.io
import hvplot.pandas # noqa
traj = [ase.io.read('/home/nicolas/Documents/Projects/glasses/Other_groups/Bennet/Gaillac2017/mqgzif4_ase.cif')]

# %%
import amof.cn as scn
rmc = scn.CoordinationNumber.from_trajectory(traj, {'Zn-N': 2.5})

# %%
path_to_gaillac_dataset = '../../data/Gaillac/dataset/'
abinitio = xr.open_dataset(path_to_gaillac_dataset+'cn.comp.nc')
abinitio = abinitio.set_index(run_id=['phase', 'mof'], append=True).sel(mof='ZIF4').coordination_number

# %%
avgglass = abinitio.sel(phase='glass').mean('run_id_level_0')
ZIF4_15glass01 = abinitio.sel(run_id_level_0 = 'ZIF4_15glass01')
crystal = abinitio.sel(phase='crystal').squeeze('run_id_level_0')
everyglass = abinitio.sel(phase='glass').rename({"run_id_level_0":"run_id"})

# %%
prop = "Zn-N"
plot = (
       everyglass.hvplot(x='Step', label='ab initio glass', by = 'run_id') * 
    avgglass.hvplot(x='Step', label='average glass')
       )
# save_hvplot(plot, f'ReaxFFinfluence_RMC_bad_{prop}')
plot

# %%
prop = "Zn-N"
print('Comparison of models in their native environment - Introduction in ReaxFF simulation')
print('Coordination number comparison for ', prop)
print("RMC initial file: ", "{:.2f}".format(rmc.cn_data[prop][0]))
print("ReaxFF:           ", "{:.2f}".format(da.coordination_number.sel(atom_pair = prop, system='rmc_bennet').dropna('Step').mean('Step').item()))
