"""
Module containing ring statistics analysis related methods

Terminology: 
    singular form 'ring' is employed by default to talk about ring statistics analysis
    plural form 'rings' or 'RINGS' refers to the Rigorous Investigation of Networks Generated using Simulations or ‘RINGS’ code
"""

import ase
import ase.data

import numpy as np
import pandas as pd
import logging
import os
import pathlib
import tempfile
import re
import joblib
import atomman
import itertools
import subprocess
import shlex
import xarray as xr

import sadi.trajectory
import sadi.atom as satom
import sadi.files.path
import sadi.pore.pysimmzeopp

logger = logging.getLogger(__name__)

class Ring(object):
    """
    Main class for ring statistics analysis analysis
    """

    def __init__(self):
        """default constructor"""
        self.ring_data = xr.DataArray(np.empty([0,0,0]), 
            coords = [('Step', np.empty([0], dtype='int64')), 
                ('ring_size', np.empty([0], dtype='int64')), 
                ('ring_var', np.empty([0], dtype='str'))], 
            name = 'ring')

    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of ring class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
        """
        ring_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        ring_class.compute_ring(trajectory, nb_set_and_cutoff, step, parallel)
        return ring_class # return class as it is a constructor

    def compute_ring(self, trajectory, nb_set_and_cutoff, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start ring analysis for volume and surfaces for %s frames", len(trajectory))
        list_of_xarray = []

        cutoff_dict = satom.format_cutoff(nb_set_and_cutoff)
        # add 0 where cutoff is not defined
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        for pair in itertools.combinations_with_replacement(atomic_numbers_unique, 2):
            if pair not in cutoff_dict.keys():
                cutoff_dict[pair] = 0

        # enlarge cutoff dict with 0 for every other values
        if parallel == False:
            for i in range(len(trajectory)):
                logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
                list_of_xarray.append(self.compute_ring_for_atom(trajectory[i], cutoff_dict))
        else:
            if type(parallel) == int:
                num_cores = parallel
            else:
                num_cores = 18 # less than 20 and nice value for 50 steps
            list_of_xarray = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_ring_for_atom)(trajectory[i], cutoff_dict) for i in range(len(trajectory)))

        dic_of_xarray = dict(zip(step, list_of_xarray))
        xa = xr.Dataset(dic_of_xarray)
        xa = xa.to_array("Step", "Step")
        xa = xr.Dataset({'ring': xa}) # one large data_array containing thermo variables
        # xa = xa.to_dataset(dim='thermo') # separate thermo variables
        self.ring_data = xa

    @staticmethod
    def read_rings_output(rstat_path):
        """
        read RINGS output from RINGS-res-N.dat whit largest possible N

        Args:
            rstat_path: pathlib path leading to rstat, containing evol-RINGS files
        """
        evol_numbers = []
        for file in rstat_path.glob('RINGS-res-*.dat'):
            evol_numbers.append(int(re.search('RINGS-res-(.*).dat', file.name).group(1)))
        N = max(evol_numbers)
        # df = pd.read_csv(rstat_path / f'evol-RINGS-{N}.dat', skiprows = list(range(1,5)), escapechar='#', sep='\s+')
        df = pd.read_csv(rstat_path / f'RINGS-res-{N}.dat', header = 1, escapechar='#', sep='\s+')
        df = df.set_index(' n')
        ar = xr.DataArray(df, dims=("ring_size", "ring_var"))
        return ar

    @staticmethod
    def fill_template(template_name, parameters, path):
        """
        Read, fill and write template

        Args:
            template_name: name of template file used in .ini file
            parameters: dic containing all the necessary input for templating
            path: pathlib path, where to write filled template under the name template_name
        """
        with open(pathlib.Path(__file__).parent.resolve() / 'template' / template_name) as f:
            template = f.read()
        script = atomman.tools.filltemplate(template, parameters, '{', '}')  
        with open(path / template_name, 'w') as f:
            f.write(script)  

    def write_input_files(self, atom, cutoff_dict, path):
        """
        Write RINGS input and option files in path

        Args:
            atom: ase atom object
            cutoff_dict: 
            path: pathlib path
        """
        parameters = {}
        parameters['number_of_atoms'] = atom.get_number_of_atoms()
        atomic_numbers_unique = list(set(atom.get_atomic_numbers()))
        elements_present_unique =  [ase.data.chemical_symbols[i] for i in atomic_numbers_unique]
        parameters['number_of_chemical_species'] = len(elements_present_unique)
        parameters['list_of_chemical_species'] = ' '.join(elements_present_unique)
        parameters['rings_maximum_search_depth_divided_by_two'] = 5 
        for i in range(3):
            parameters[f"cell{'abc'[i]}"] = str(atom.cell[i])[1:-1]
        parameters['cutoff_lines'] = ''
        for key, value in cutoff_dict.items():
            line_list = [ase.data.chemical_symbols[i] for i in key] + [str(value), """# \n """]
            parameters['cutoff_lines'] += ' '.join(line_list)
        parameters['Grtot'] = max(cutoff_dict.values())
        self.fill_template('input.inp', parameters, path)
        self.fill_template('options', parameters, path)

    def compute_ring_for_atom(self, atom, cutoff_dict):
        """
        Args:
            atom: ase atom object
        Returns:
            dic: dictionary with output from zeopp vol and sa
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirpath = pathlib.Path(tmpdirname)
            (tmpdirpath / 'data').mkdir(parents=True, exist_ok=True)
            atom.write(tmpdirname + '/data/atom.xyz')
            self.write_input_files(atom, cutoff_dict, tmpdirpath)
            
            # warning, make sure no user input is possible inside arg. 
            # Here shlex.quote is used as precaution in the only possible variable of the string
            arg = f"cd {shlex.quote(tmpdirname)} && rings input.inp"
            p = subprocess.Popen(arg, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
            p.wait()

            # arg_list = shlex.split(arg)  # if no need to use shell=True                            
            # p = Popen(arg_list, stdin=PIPE, stdout=PIPE, stderr=PIPE) 


            # os.system(f"cd {tmpdirname} && rings input.inp")            

            ring_ar = self.read_rings_output(tmpdirpath / 'rstat')
            a=1
        return ring_ar

    def write_to_file(self, filename):
        """path_to_output: where the ring object will be written"""
        filename = sadi.files.path.append_suffix(filename, 'ring')
        self.surface_volume.to_feather(filename)

    @classmethod
    def from_file(cls, filename):
        """
        constructor of ring class from ring file
        """
        ring_class = cls() # initialize class
        ring_class.read_surface_volume_file(filename)
        return ring_class # return class as it is a constructor

    def read_ring_file(self, filename):
        """path_to_data: where the ring object is"""
        filename = sadi.files.path.append_suffix(filename, 'ring')
        self.surface_volume = pd.read_feather(filename)
